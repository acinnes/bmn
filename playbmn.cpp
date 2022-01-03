#include <iostream>
#include <random>
#include <utility>
#include <cassert>
#include <ctime>
#include <cstdio>
#include <future>
#include <mutex>

// Generic definitions to support using classes in CUDA kernels
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <nvrtc_helper.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#endif // USE_CUDA

bool verbose = false;

// Beggar my neighbour deal explorer.
//
// Efficiently search (in pseudo random order) through shuffled deck starting points,
// to find the longest possible games of BMN.

// Use a simplified representation of cards that is optimized for BMN.
enum Card
{
    numeral = 0,
    jack = 1,
    queen = 2,
    king = 3,
    ace = 4
};

// Representation of a stack of cards optimized for BMN, specifically designed for efficiently popping
// from the beginning of the stack and appending to the end.
// Takes advantage of the fact that a stack can contain at most all 52 cards in a standard deck.
// Use a power of two sized circular buffer with two index values for popping and appending.
// Mask the index values when referencing, and just increment indefinitely when popping and appending,
// in order to not waste cycles with unnecessary branches for handling wrap-around.

class StackOfCards
{
  private:
    unsigned first = 0;
    unsigned insert = 0;
    Card cards[64];
    const unsigned stack_mask = sizeof(cards)/sizeof(Card) - 1;

public:
    CUDA_CALLABLE StackOfCards() = default;

    // Initialize the stack with the full standard deck, in a canonical sort order.
    CUDA_CALLABLE void set_full_deck()
        {
            unsigned index = 0;
            for (int i = 0; i < 4; i++)
            {
                cards[index++] = ace;
                cards[index++] = king;
                cards[index++] = queen;
                cards[index++] = jack;
                for (int j = 0; j < 9; j++)
                {
                    cards[index++] = numeral;
                }
            }
            insert = index;
        }

    CUDA_CALLABLE std::string to_string() {
        const char syms[] = {'-', 'J', 'Q', 'K', 'A'};
        char tmp[53];
        char *t = tmp;
        assert(num_cards() <= 52);
        for (unsigned i = first; i < insert; i++) {
            auto card = cards[i & stack_mask];
            assert(card <= ace);
            *t++ = syms[card];
        }
        *t = 0;
        return std::string(tmp);
    }

    CUDA_CALLABLE void swap(unsigned i, unsigned j) {
        assert(i < num_cards());
        assert(j < num_cards());
        std::swap(cards[(first+i) & stack_mask], cards[(first+j) & stack_mask]);
    }

    CUDA_CALLABLE Card pop() {
        assert(num_cards() <= 52);
        assert(num_cards() > 0);
        return cards[(first++) & stack_mask];
    }

    CUDA_CALLABLE void append(Card c) {
        assert(num_cards() < 52);
        cards[(insert++) & stack_mask] = c;
    }

    // Pick up a stack of cards (append here and leave the source stack empty)
    CUDA_CALLABLE void pick_up(StackOfCards& s) {
        for (unsigned i = s.first; i < s.insert; i++) {
            append(s.cards[i & stack_mask]);
        }
        s.insert = s.first;
    }

    CUDA_CALLABLE unsigned num_cards() {
        return insert - first;
    }

    CUDA_CALLABLE bool not_empty() {
        return insert != first;
    }

    // Initialize the two hands from a starting deck
    CUDA_CALLABLE void set_hands(StackOfCards& a, StackOfCards& b) {
        assert(num_cards() == 52);
        // put first half of full deck in a, second half in b
        for (int i=0; i<26; i++) {
            a.cards[i] = cards[i];
        }
        a.first = 0;
        a.insert = 26;
        for (int i=0; i<26; i++) {
            b.cards[i] = cards[i+26];
        }
        b.first = 0;
        b.insert = 26;
    }
};

// Play out a deal, reporting the number of turns played in the game.
// Based on https://github.com/matthewmayer/beggarmypython/blob/master/beggarmypython/__init__.py
CUDA_CALLABLE void play(StackOfCards& deal, unsigned& turns, unsigned& tricks)
{
    turns = 0;
    tricks = 0;
    StackOfCards hands[2];
    StackOfCards pile;
    deal.set_hands(hands[0], hands[1]);
    unsigned player = 0;

    if (verbose) {
        std::string a = hands[0].to_string();
        std::string b = hands[1].to_string();
        printf("Starting hands: %s/%s\n", a.data(), b.data());
    }

    while (hands[0].not_empty() && hands[1].not_empty()) {
        bool battle_in_progress = false;
        unsigned cards_to_play = 1;
        while (cards_to_play > 0) {
#ifdef EXTRA_VERBOSE
            if (verbose) {
                std::string a = hands[0].to_string();
                std::string b = hands[1].to_string();
                std::string p = pile.to_string();
                printf("Turn %d: %s/%s/%s\n", turns, a.data(), b.data(), p.data());
            };
#endif
            Card next_card;
            assert(player < 2);
            if (hands[player].not_empty()) {
                next_card = hands[player].pop();
            } else {
                break;
            }
            turns++;
            pile.append(next_card);
            if (next_card == numeral) {
                if (battle_in_progress) {
                    cards_to_play--;
                } else {
                    player ^= 1;
                }
            } else {
                battle_in_progress = true;
                cards_to_play = (unsigned) next_card;
                player ^= 1;
            }
        }

        tricks++;
        player ^= 1;
        hands[player].pick_up(pile);
        if (verbose) {
            std::string a = hands[0].to_string();
            std::string b = hands[1].to_string();
            std::string p = pile.to_string();
            printf("Trick %d: %s/%s\n", tricks, a.data(), b.data());
        }
    }
}

std::mutex printf_mutex;
bool progress_printed = false;

class BestDealSearcher {
private:
    clock_t start_time;
    clock_t last_print_time;
    unsigned best_turns = 0;
    unsigned best_tricks = 0;
    unsigned long deals_tested = 0;
    int threadid = 0;
    StackOfCards deck;
    std::mt19937_64 rng;

public:
    CUDA_CALLABLE BestDealSearcher(int seed) {
        threadid = seed;
        init(seed);
        start_time = last_print_time = clock();
    }

    CUDA_CALLABLE void track_best_deal(StackOfCards& deal) {
        unsigned turns, tricks;
        play(deal, turns, tricks);
        ++deals_tested;
        auto now = clock();
        if (threadid <= 1 && (now - last_print_time) / CLOCKS_PER_SEC >= 1) {
            last_print_time = now;
            float secs_since_start = (now - start_time) / CLOCKS_PER_SEC;
            printf("\r%g second, %ld deals tested (%g per second)",
                   secs_since_start, deals_tested, deals_tested / secs_since_start);
            progress_printed = true;
        }
        if (turns > best_turns || tricks > best_tricks) {
            const std::lock_guard<std::mutex> lock(printf_mutex);
            if (progress_printed) {
                printf("\n");
                progress_printed = false;
            }
            printf("%s: %d turns, %d tricks\n", deal.to_string().data(), turns, tricks);
            if (turns > best_turns)
                best_turns = turns;
            if (tricks > best_tricks)
                best_tricks = tricks;
        }
    }

    CUDA_CALLABLE void init(int seed) {
        // Start with a fixed per-suit descending order, for reproducibilty.
        deck.set_full_deck();
        // If seed is 0, leave rng in default initial state for reproducibilty
        if (seed != 0) {
            std::default_random_engine seeder(seed);
            std::seed_seq mt_seed{seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), };
            std::mt19937_64 new_rng(mt_seed);
            rng = new_rng;
        }
    }

    CUDA_CALLABLE void search(unsigned iterations) {
        // Conduct pseudo-random search for a deal that produces the longest game. The basic
        // approach is to start with a default sorted deck state, and then incrementally shuffle
        // this to generate deals which are different from ones we've already tested. We want this
        // to be very efficient, and as guaranteed as possible to not repeat previously seen deals,
        // or at least to not get stuck in a loop that won't explore new possible deals.
        //
        // It is well-known that a simple random shuffle (linear scan with pair swapping among the
        // remaining elements) will provide an unbiased random selection of a new permutation. So
        // this will be used here, except we will also test games at all of the intermediate swap
        // points. A similar approach could be used where a simple rotating position is used as the
        // swap target, with the other element chosen at random.

        auto num_cards = deck.num_cards();
        bool keep_going = true;
        while (keep_going) {
            for (unsigned i=0; i<50; i++) {
                std::uniform_int_distribution<unsigned> u(0, num_cards-i-1);
                unsigned j = i + u(rng);
                deck.swap(i, j);
                track_best_deal(deck);
                if (iterations-- == 0)
                    keep_going = false;
            }
        }
    }
};

CUDA_CALLABLE void run_search(int seed) {
    BestDealSearcher searcher(seed);
    while (true)
        searcher.search(1e6);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        // Single-threaded search mode from fixed seed.
        BestDealSearcher searcher(0);
        while (true)
            searcher.search(1e6);

    } else if (strlen(argv[1]) >= 52) {
        // Test mode, to ensure accurate match of Python reference implementation.
        // Expect deal string in format used by https://github.com/matthewmayer/beggarmypython
        StackOfCards deal;
        for (char *p=argv[1]; *p != '\0'; p++) {
            int c = *p;
            if (c == '-') {
                deal.append(numeral);
            } else if (c == 'A') {
                deal.append(ace);
            } else if (c == 'K') {
                deal.append(king);
            } else if (c == 'Q') {
                deal.append(queen);
            } else if (c == 'J') {
                deal.append(jack);
            } else if (c == '/') {
                // ignore the hand separator char
            } else {
                std::cerr << "Invalid card: " << c << std::endl;
                return 1;
            }
        }

        if (argc > 2)
            verbose = true;
        unsigned turns, tricks;
        play(deal, turns, tricks);
        printf("There were %d turns\n", turns);
        printf("There were %d tricks\n", tricks);

    } else if (strcmp(argv[1], "-t") == 0) {
        int threads = atoi(argv[2]);
        // Multi-threaded searching with different seeds
        std::vector<std::future<void>> worker_futures;
        for (int i=1; i<=threads; i++) {
            worker_futures.push_back(std::async(std::launch::async, run_search, i));
        }
        for (auto&& worker_future : worker_futures) {
            worker_future.get();
        }
    }
    return 0;
}
