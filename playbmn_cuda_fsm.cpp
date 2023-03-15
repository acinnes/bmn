#include <iostream>
#include <random>
#include <utility>
#include <cassert>
#include <ctime>
#include <cstdio>
#include <future>
#include <mutex>

// This version is designed to only run on CUDA.

// Generic definitions to support using classes in CUDA kernels
#define DEVICE_CALLABLE __host__ __device__
#define DEVICE_ONLY  __device__

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <nvrtc_helper.h>
// helper functions and utilities to work with CUDA
//#include <helper_functions.h>

class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void *operator new[](size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }
    
    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }

    void operator delete[](void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

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
    ace = 4,
    no_card = 5,
    NUM_CARDS = 6
};

enum Player
{
    player_0 = 0,
    player_1 = 1,
    discard_pile = 2,
    NUM_PLAYERS = 3
};

enum GameState
{
    discard = 0,
    pay1 = 1,
    pay2 = 2,
    pay3 = 3,
    pay4 = 4,
    pickup_by_0 = 5,
    pickup_by_1 = 6,
    game_over = 7,
    logic_error = 8,
    NUM_STATES = 7              // game_over and logic_error are not used for lookups
};

// Action directions for each cycle of the game loop, to direct side effects and update the game
// state. The relevant actions for each game state are encoded into a lookup table, indexed by
// current player, game_state, and next_card. The action information for that combination takes care
// of encoded whether we need to switch player, whether a "battle" is in progress and for how many
// moves, whether a turn should be counted and whether a trick should be counted.
struct NextAction
{
    Player destination;         // index to the right stack of cards to receive next_card
    Player next_player;         // player (or pile as a psuedo-player) who takes the next turn
    GameState next_state;       // game state facing the next player
    bool count_turn;            // true (ie. 1) if turn count should be incremented
    bool count_trick;           // same for trick count
};

// Action lookup table to encode all of game state transitions so that the main game loop has as
// little branching as possible, to minimise divergence across CUDA threads playing independent
// games in parallel. The key state representation decision which enables this is that the pile is
// treated as a pseudo player, so the core game loop is always has one player playing their next
// card by adding it to another player's stack. This either the real players taking turns as
// required to add a card to the discard pile, or the discard pile "playing" all of its cards
// one-by-one (in separate turns) to the trick winner. To avoid counting the discard pile's turns,
// we make the turn and trick increments for each game cycle part of the action lookup information.

DEVICE_ONLY
NextAction action_table[NUM_PLAYERS][NUM_STATES][NUM_CARDS] =
{
    // Inner elements are of the form { destination, next_player, next_state, count_turn, count_trick }

    // PLAYER = player_0
    {
        // state = discard
        {
            { discard_pile, player_1, discard, 1, 0 },  // card = numeral
            { discard_pile, player_1, pay1, 1, 0 },     // card = jack
            { discard_pile, player_1, pay2, 1, 0 },     // card = queen
            { discard_pile, player_1, pay3, 1, 0 },     // card = king
            { discard_pile, player_1, pay4, 1, 0 },     // card = ace
            { discard_pile, player_1, game_over, 0, 1 } // card = no_card
        },
        // state = pay1
        {
            { discard_pile, discard_pile, pickup_by_1, 1, 1 },
            { discard_pile, player_1, pay1, 1, 0 },
            { discard_pile, player_1, pay2, 1, 0 },
            { discard_pile, player_1, pay3, 1, 0 },
            { discard_pile, player_1, pay4, 1, 0 },
            { discard_pile, player_1, game_over, 0, 1 }
        },
        // state = pay2
        {
            { discard_pile, player_0, pay1, 1, 0 },
            { discard_pile, player_1, pay1, 1, 0 },
            { discard_pile, player_1, pay2, 1, 0 },
            { discard_pile, player_1, pay3, 1, 0 },
            { discard_pile, player_1, pay4, 1, 0 },
            { discard_pile, player_1, game_over, 0, 1 }
        },
        // state = pay3
        {
            { discard_pile, player_0, pay2, 1, 0 },
            { discard_pile, player_1, pay1, 1, 0 },
            { discard_pile, player_1, pay2, 1, 0 },
            { discard_pile, player_1, pay3, 1, 0 },
            { discard_pile, player_1, pay4, 1, 0 },
            { discard_pile, player_1, game_over, 0, 1 }
        },
        // state = pay4
        {
            { discard_pile, player_0, pay3, 1, 0 },
            { discard_pile, player_1, pay1, 1, 0 },
            { discard_pile, player_1, pay2, 1, 0 },
            { discard_pile, player_1, pay3, 1, 0 },
            { discard_pile, player_1, pay4, 1, 0 },
            { discard_pile, player_1, game_over, 0, 1 }
        },
        // state = pickup_by_0
        {
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 }
        },
        // state = pickup_by_1
        {
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 },
            { player_0, player_0, logic_error, 0, 0 }
        }
    },

    // PLAYER = player_1
    {
        // state = discard
        {
            { discard_pile, player_0, discard, 1, 0 },  // card = numeral
            { discard_pile, player_0, pay1, 1, 0 },     // card = jack
            { discard_pile, player_0, pay2, 1, 0 },     // card = queen
            { discard_pile, player_0, pay3, 1, 0 },     // card = king
            { discard_pile, player_0, pay4, 1, 0 },     // card = ace
            { discard_pile, player_0, game_over, 0, 1 } // card = no_card
        },
        // state = pay1
        {
            { discard_pile, discard_pile, pickup_by_0, 1, 1 },
            { discard_pile, player_0, pay1, 1, 0 },
            { discard_pile, player_0, pay2, 1, 0 },
            { discard_pile, player_0, pay3, 1, 0 },
            { discard_pile, player_0, pay4, 1, 0 },
            { discard_pile, player_0, game_over, 0, 1 }
        },
        // state = pay2
        {
            { discard_pile, player_1, pay1, 1, 0 },
            { discard_pile, player_0, pay1, 1, 0 },
            { discard_pile, player_0, pay2, 1, 0 },
            { discard_pile, player_0, pay3, 1, 0 },
            { discard_pile, player_0, pay4, 1, 0 },
            { discard_pile, player_0, game_over, 0, 1 }
        },
        // state = pay3
        {
            { discard_pile, player_1, pay2, 1, 0 },
            { discard_pile, player_0, pay1, 1, 0 },
            { discard_pile, player_0, pay2, 1, 0 },
            { discard_pile, player_0, pay3, 1, 0 },
            { discard_pile, player_0, pay4, 1, 0 },
            { discard_pile, player_0, game_over, 0, 1 }
        },
        // state = pay4
        {
            { discard_pile, player_1, pay3, 1, 0 },
            { discard_pile, player_0, pay1, 1, 0 },
            { discard_pile, player_0, pay2, 1, 0 },
            { discard_pile, player_0, pay3, 1, 0 },
            { discard_pile, player_0, pay4, 1, 0 },
            { discard_pile, player_0, game_over, 0, 1 }
        },
        // state = pickup_by_0
        {
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 }
        },
        // state = pickup_by_1
        {
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 },
            { player_1, player_1, logic_error, 0, 0 }
        }
    },

    // PLAYER = discard_pile
    {
        // state = discard
        {
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 }
        },
        // state = pay1
        {
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 }
        },
        // state = pay2
        {
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 }
        },
        // state = pay3
        {
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 }
        },
        // state = pay4
        {
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 },
            { discard_pile, discard_pile, logic_error, 0, 0 }
        },
        // state = pickup_by_0
        {
            { player_0, discard_pile, pickup_by_0, 0, 0 },
            { player_0, discard_pile, pickup_by_0, 0, 0 },
            { player_0, discard_pile, pickup_by_0, 0, 0 },
            { player_0, discard_pile, pickup_by_0, 0, 0 },
            { player_0, discard_pile, pickup_by_0, 0, 0 },
            { player_0, player_0, discard, 0, 0 }
        },
        // state = pickup_by_1
        {
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, player_1, discard, 0, 0 }
        }
    }
};


// Representation of a stack of cards optimized for BMN, specifically designed for efficiently popping
// from the beginning of the stack and appending to the end.
// Takes advantage of the fact that a stack can contain at most all 52 cards in a standard deck.
// Use a power of two sized circular buffer with two index values for popping and appending.
// Mask the index values when referencing, and just increment indefinitely when popping and appending,
// in order to not waste cycles with unnecessary branches for handling wrap-around.

class StackOfCards : public Managed
{
  private:
    unsigned first = 0;
    unsigned insert = 0;
    Card cards[64];
    static const unsigned stack_mask = sizeof(cards)/sizeof(Card) - 1;

public:
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
    void set_full_deck() {
        unsigned index = 0;
        for (int i = 0; i < 4; i++) {
            cards[index++] = ace;
            cards[index++] = king;
            cards[index++] = queen;
            cards[index++] = jack;
            for (int j = 0; j < 9; j++) {
                cards[index++] = numeral;
            }
        }
        insert = index;
    }

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
    bool test_and_swap(unsigned i, unsigned j) {
        assert(first == 0);
        assert(i < num_cards());
        assert(j < num_cards());
        // Can omit the offset and mask operations in this special case.
        auto ival = cards[i];
        auto jval = cards[j];
        if (ival != jval) {
            cards[i] = jval;
            cards[j] = ival;
            return true;
        }
        return false;
    }

    DEVICE_CALLABLE
    void swap(unsigned i, unsigned j) {
        assert(first == 0);
        assert(i < num_cards());
        assert(j < num_cards());
        // Can omit the offset and mask operations in this special case.
        auto ival = cards[i];
        auto jval = cards[j];
        cards[i] = jval;
        cards[j] = ival;
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

// Play out a deal using finite state machine approach, to provide better concurrency in CUDA warps.
DEVICE_CALLABLE
void play_fsm(StackOfCards& deal, unsigned& turns, unsigned& tricks) {
    turns = 0;
    tricks = 0;
    StackOfCards hands[NUM_PLAYERS];
    deal.set_hands(hands[0], hands[1]);
    Player player = player_0;
    GameState game_state = discard;
    Card next_card;

    while (game_state != game_over) {
        if (hands[player].not_empty()) {
            next_card = hands[player].pop();
        } else {
            next_card = no_card;
        }
        NextAction &next_action = action_table[player][game_state][next_card];
        assert(next_action.next_state != logic_error);
        if (next_card != no_card)
            hands[next_action.destination].append(next_card);
        player = next_action.next_player;
        game_state = next_action.next_state;
        turns += next_action.count_turn;
        tricks += next_action.count_trick;
    }
}

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
#ifdef EXTRA_VERBOSE
#ifndef __CUDACC__
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

class BestDealSearcher : public Managed {
public:
    unsigned best_turns = 0;
    unsigned best_tricks = 0;
    unsigned long deals_tested = 0;
    StackOfCards deck;
    StackOfCards best_turns_deal;
    StackOfCards best_tricks_deal;
    // Kernel version of curand
    curandState rng;

    BestDealSearcher() = default;

    DEVICE_CALLABLE
    void track_best_deal(StackOfCards& deal) {
        unsigned turns, tricks;
        play_fsm(deal, turns, tricks);
        ++deals_tested;
        if (turns > best_turns) {
            best_turns = turns;
            best_turns_deal = deal;
        }
        if (tricks > best_tricks) {
            best_tricks = tricks;
            best_tricks_deal = deal;
        }
    }

    // This can only be called from cuda kernel, not host code.
    DEVICE_ONLY
    void init(int seed, int sequence) {
        // Start with a fixed per-suit descending order, for reproducibilty.
        deck.set_full_deck();
        curand_init(seed, sequence, 0, &rng);
    }

    // This can only be called from cuda kernel, not host code.
    DEVICE_ONLY
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

#ifdef USE_CUDA_SHARED
        int t = threadIdx.x;
        extern __shared__ StackOfCards local_deck[];
        local_deck[t] = deck;
#define DECK local_deck[t]
#else
#define DECK deck
#endif
        auto num_cards = deck.num_cards();
        bool keep_going = true;
        while (keep_going) {
            for (unsigned i=0; i<num_cards-1; i++) {
                // Ignore the slight bias from using simple modulus on random unsigned
                unsigned offset = curand(&rng) % (num_cards-i);
                unsigned j = i + offset;
                DECK.swap(i, j);
            }
            track_best_deal(DECK);
            if (iterations-- == 0)
                keep_going = false;
        }
#ifdef USE_CUDA_SHARED
        deck = local_deck[t];
#endif
    }
};

// ------------------------------------------------------------------------------------------
// Controller thread to report progress as potentially parallel search is in progress. One host
// thread passes work to a cuda stream to do a decent number of iterations, then processes the
// results. If the cuda stream is used as a filter to identify deals that don't complete within a
// threshold number of turns, then the host thread could work on playing each unfinished deal to get
// the final result, after dispatching the cuda stream to work on another set of games.

// Based on experimentation on laptop with GTX 1650, 1 block with 1 thread does ~20k checks per sec,
// and throughput plateaus by the time we have 128 blocks with 1 thread each.
#define BLOCKS_PER_WORKER 128
#define THREADS_PER_BLOCK 32
#define SEARCHERS_PER_WORKER (BLOCKS_PER_WORKER * THREADS_PER_BLOCK)

unsigned blocks = BLOCKS_PER_WORKER;
unsigned threads = THREADS_PER_BLOCK;
unsigned num_searchers = SEARCHERS_PER_WORKER;

__global__
void cuda_init_searchers(BestDealSearcher *searchers) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    searchers[tid].init(blockIdx.x, threadIdx.x);
}

__global__
void cuda_run_searchers(BestDealSearcher *searchers) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // run a decent number of iterations, taking on the order of at most a few seconds, before
    // stopping to update and report on the global progress.
    searchers[tid].search((unsigned)10);
}

std::mutex global_mutex;
bool progress_printed = false;
unsigned global_best_turns = 0;
unsigned global_best_tricks = 0;
unsigned long global_deals_tested = 0;

void run_search(int worker_id) {
    // Generalize to allow a list of searchers to be run/managed by each worker (since in cuda mode
    // we may want to have multiple thread blocks and/or multiple threads per block, depending how
    // badly the play function diverges in a warp of threads).
    BestDealSearcher *searchers = new BestDealSearcher[num_searchers];

    cuda_init_searchers<<<blocks, threads>>>(searchers);
    cudaDeviceSynchronize();
    clock_t start_time = clock();
    clock_t last_update_time = start_time;

    while (true) {
#ifdef USE_CUDA_SHARED
        cuda_run_searchers<<<blocks, threads, threads*sizeof(StackOfCards)>>>(searchers);
#else
        cuda_run_searchers<<<blocks, threads>>>(searchers);
#endif
        cudaDeviceSynchronize();

        // report progress and best deal so far
        const std::lock_guard<std::mutex> lock(global_mutex);
        auto now = clock();
        if ((now - last_update_time) / CLOCKS_PER_SEC >= 1) {
            last_update_time = now;
            for (unsigned i=0; i<num_searchers; i++) {
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
    const char *ev;
    ev = getenv("SEARCHERS_PER_WORKER");
    if (ev) {
        num_searchers = atoi(ev);
        blocks = 0;
    }
    ev = getenv("THREADS_PER_BLOCK");
    if (ev) {
        threads = atoi(ev);
        blocks = 0;
    }
    if (blocks == 0) {
        blocks = num_searchers / threads;
    }
    printf("%d/%d blocks/threads == %d searchers\n", blocks, threads, num_searchers);
    printf("sizeof(BestDealSearcher) is %zd bytes\n", sizeof(BestDealSearcher));
    printf("sizeof(StackOfCards) is %zd bytes\n", sizeof(StackOfCards));
    printf("sizeof(action_table) is %zd bytes\n", sizeof(action_table));

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

        bool test_fsm = false;
        if (argc > 2) {
            verbose = true;
            if (strcmp(argv[2], "test-fsm") == 0) {
                test_fsm = true;
            }
        }

        unsigned turns, tricks;
        if (test_fsm) {
            printf("Testing play_fsm...\n");
            play_fsm(deal, turns, tricks);
            printf("There were %d turns\n", turns);
            printf("There were %d tricks\n", tricks);
            verbose = false;
            printf("Testing original play...\n");
        }
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
