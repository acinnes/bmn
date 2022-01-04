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
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
//#include <nvrtc_helper.h>
// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
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
    static const unsigned stack_mask = sizeof(cards)/sizeof(Card) - 1;

public:
    DEVICE_CALLABLE
    StackOfCards() = default;

    DEVICE_CALLABLE
    bool operator==(const StackOfCards& other) {
        return (first == other.first) && (insert == other.insert)
            && std::memcmp(cards, other.cards, sizeof(cards)) == 0;
    }

    DEVICE_CALLABLE
    bool operator!=(const StackOfCards& other) { return !(this->operator==(other)); }

    DEVICE_CALLABLE
    bool operator[](const int i) { return cards[(i+first) & stack_mask]; }

    // Initialize the stack with the full standard deck, in a canonical sort order.
    DEVICE_CALLABLE
    void set_full_deck()
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

    DEVICE_CALLABLE
    std::string to_string() {
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

    DEVICE_CALLABLE
    bool swap(unsigned i, unsigned j) {
        assert(i < num_cards());
        assert(j < num_cards());
        auto ival = cards[(first+i) & stack_mask];
        auto jval = cards[(first+j) & stack_mask];
        if (ival != jval) {
            cards[(first+i) & stack_mask] = jval;
            cards[(first+j) & stack_mask] = ival;
            return true;
        }
        return false;
    }

    DEVICE_CALLABLE
    Card pop() {
        assert(num_cards() <= 52);
        assert(num_cards() > 0);
        return cards[(first++) & stack_mask];
    }

    DEVICE_CALLABLE
    void append(Card c) {
        assert(num_cards() < 52);
        cards[(insert++) & stack_mask] = c;
    }

    // Pick up a stack of cards (append here and leave the source stack empty)
    DEVICE_CALLABLE
    void pick_up(StackOfCards& s) {
        for (unsigned i = s.first; i < s.insert; i++) {
            append(s.cards[i & stack_mask]);
        }
        s.insert = s.first;
    }

    DEVICE_CALLABLE
    unsigned num_cards() {
        return insert - first;
    }

    DEVICE_CALLABLE
    bool not_empty() {
        return insert != first;
    }

    // Initialize the two hands from a starting deck
    DEVICE_CALLABLE
    void set_hands(StackOfCards& a, StackOfCards& b) {
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
DEVICE_CALLABLE
void play(StackOfCards& deal, unsigned& turns, unsigned& tricks) {
    turns = 0;
    tricks = 0;
    StackOfCards hands[2];
    StackOfCards pile;
    deal.set_hands(hands[0], hands[1]);
    unsigned player = 0;

#ifndef __CUDACC__
    if (verbose) {
        std::string a = hands[0].to_string();
        std::string b = hands[1].to_string();
        printf("Starting hands: %s/%s\n", a.data(), b.data());
    }
#endif

    while (hands[0].not_empty() && hands[1].not_empty()) {
        bool battle_in_progress = false;
        unsigned cards_to_play = 1;
        while (cards_to_play > 0) {
#ifndef __CUDACC__
#ifdef EXTRA_VERBOSE
            if (verbose) {
                std::string a = hands[0].to_string();
                std::string b = hands[1].to_string();
                std::string p = pile.to_string();
                printf("Turn %d: %s/%s/%s\n", turns, a.data(), b.data(), p.data());
            };
#endif
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
#ifndef __CUDACC__
        if (verbose) {
            std::string a = hands[0].to_string();
            std::string b = hands[1].to_string();
            std::string p = pile.to_string();
            printf("Trick %d: %s/%s\n", tricks, a.data(), b.data());
        }
#endif
    }
}

#ifdef USE_CUDA
class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};
#else
class Managed {};
#endif

class BestDealSearcher : Managed {
public:
    unsigned best_turns = 0;
    unsigned best_tricks = 0;
    unsigned long deals_tested = 0;
    StackOfCards deck;
    StackOfCards best_turns_deal;
    StackOfCards best_tricks_deal;
#ifdef __CUDACC__XXX
    curand rng;
#else
    std::mt19937_64 rng;
#endif

    DEVICE_CALLABLE
    BestDealSearcher() = default;

    DEVICE_CALLABLE
    void track_best_deal() {
        unsigned turns, tricks;
        play(deck, turns, tricks);
        ++deals_tested;
        if (turns > best_turns) {
            best_turns = turns;
            best_turns_deal = deck;
        }
        if (tricks > best_tricks) {
            best_tricks = tricks;
            best_tricks_deal = deck;
        }
    }

    DEVICE_CALLABLE
    void init(int seed) {
        // Start with a fixed per-suit descending order, for reproducibilty.
        deck.set_full_deck();
#ifdef __CUDACC__XXX
#else
        // If seed is 0, leave rng in default initial state for reproducibilty
        if (seed != 0) {
            std::default_random_engine seeder(seed);
            std::seed_seq mt_seed{seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), };
            std::mt19937_64 new_rng(mt_seed);
            rng = new_rng;
        }
#endif
    }

    DEVICE_CALLABLE
    void search(unsigned iterations) {
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
            for (unsigned i=0; i<num_cards-1; i++) {
                std::uniform_int_distribution<unsigned> u(0, num_cards-i-1);
                unsigned j = i + u(rng);
                if (deck.swap(i, j)) {
                    track_best_deal();
                    if (iterations-- == 0)
                        keep_going = false;
                }
            }
        }
    }
};

// ------------------------------------------------------------------------------------------
// Controller threads to report progress as potentially parallel search is in progress. When
// USE_CUDA is on, each host thread passes work to a cuda stream to do a decent number of
// iterations, then processes the results. If the cuda stream is used as a filter to identify deals
// that don't complete within a threshold number of turns, then the host thread will work on playing
// each unfinished deal to get the final result. Each stream will internally run threads and thread
// blocks in parallel to the extent that is effective, and multiple streams can operate in parallel.

#ifdef __CUDACC__
__global__
void cuda_searcher(BestDealSearcher *searcher) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    searcher.init(tid);
    // run a decent number of iterations, taking on the order of at most a few seconds, before
    // stopping to update and report on the global progress.
    searcher.search(1e5);
}
#endif

std::mutex global_mutex;
bool progress_printed = false;
unsigned global_best_turns = 0;
unsigned global_best_tricks = 0;
unsigned long global_deals_tested = 0;

void run_search(int worker_id) {
    // Generalize to allow a list of searchers to be run/managed by each worker (since in cuda mode
    // we may want to have multiple thread blocks and/or multiple threads per block, depending how
    // badly the play function diverges in a warp of threads).
#ifdef USE_CUDA
#define SEARCHERS_PER_WORKER 1
#else
#define SEARCHERS_PER_WORKER 1
#endif
    BestDealSearcher *searchers = new BestDealSearcher[SEARCHERS_PER_WORKER];
    for (int i=0; i<SEARCHERS_PER_WORKER; i++)
        searchers[i].init(worker_id * SEARCHERS_PER_WORKER + i);

    clock_t start_time = clock();
    clock_t last_update_time = start_time;
    while (true) {
        // run a decent number of iterations, taking on the order of a few seconds, before stopping
        // to update and report on the global progress.
        for (int i=0; i<SEARCHERS_PER_WORKER; i++)
            searchers[i].search((unsigned)1e5);

        // report progress and best deal so far
        const std::lock_guard<std::mutex> lock(global_mutex);
        auto now = clock();
        if ((now - last_update_time) / CLOCKS_PER_SEC >= 1) {
            last_update_time = now;
            for (int i=0; i<SEARCHERS_PER_WORKER; i++) {
                auto& searcher = searchers[i];
                global_deals_tested += searcher.deals_tested;
                searcher.deals_tested = 0;
                if (worker_id <= 1) {
                    double secs_since_start = (now - start_time) / CLOCKS_PER_SEC;
                    printf("\r%g seconds, %ld deals tested (%g per second)",
                           secs_since_start, global_deals_tested, global_deals_tested / secs_since_start);
                    progress_printed = true;
                }

                auto best_turns = searcher.best_turns;
                auto best_tricks = searcher.best_tricks;
                if (best_turns > global_best_turns || best_tricks > global_best_tricks) {
                    if (progress_printed) {
                        printf("\n");
                        progress_printed = false;
                    }
                    if (best_turns > global_best_turns) {
                        global_best_turns = best_turns;
                        auto& deck = searcher.best_turns_deal;
                        unsigned turns, tricks;
                        play(deck, turns, tricks);
                        printf("[T%d] %s: %d turns, %d tricks\n", worker_id, deck.to_string().data(), turns, tricks);
                    }
                    if (best_tricks > global_best_tricks) {
                        global_best_tricks = best_tricks;
                        if (searcher.best_tricks_deal != searcher.best_turns_deal) {
                            auto& deck = searcher.best_tricks_deal;
                            unsigned turns, tricks;
                            play(deck, turns, tricks);
                            printf("[T%d] %s: %d turns, %d tricks\n", worker_id, searcher.best_tricks_deal.to_string().data(), turns, tricks);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc == 1) {
        // Single-threaded search mode from fixed seed.
        run_search(0);

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
