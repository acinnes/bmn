#ifndef OVERRIDES
#define USE_SHARED_MEM_FOR_DEALS
#define COMPACT_DECK
#define PACKED_CARDS
#define PACKED_ACTIONS
#endif


#include <iostream>
#include <random>
#include <utility>
#include <cassert>
#include <ctime>
#include <cstdio>
#include <future>
#include <mutex>

// Generic definitions to support using classes in CUDA kernels
#ifdef USE_CUDA
#define DEVICE_CALLABLE __host__ __device__
#define DEVICE_ONLY  __device__
#define CONSTANT __constant__
#else
#define DEVICE_CALLABLE
#define DEVICE_ONLY
#define CONSTANT
#endif

#ifdef USE_CUDA
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
#else
class Managed {};
#endif // USE_CUDA


bool verbose = false;

// Beggar my neighbour deal explorer.
//
// Efficiently search (in pseudo random order) through shuffled deck starting points,
// to find the longest possible games of BMN.

// Use a simplified representation of cards that is optimized for BMN.
#ifdef PACKED_CARDS
enum Card : uint8_t
#else
enum Card
#endif
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
#ifdef PACKED_ACTIONS
struct NextAction
{
    Player destination : 4;         // index to the right stack of cards to receive next_card
    Player next_player : 2;         // player (or pile as a psuedo-player) who takes the next turn
    GameState next_state : 4;       // game state facing the next player
    bool count_turn : 1;            // true (ie. 1) if turn count should be incremented
    bool count_trick : 1;           // same for trick count
};
#else
struct NextAction
{
    Player destination;         // index to the right stack of cards to receive next_card
    Player next_player;         // player (or pile as a psuedo-player) who takes the next turn
    GameState next_state;       // game state facing the next player
    bool count_turn;            // true (ie. 1) if turn count should be incremented
    bool count_trick;           // same for trick count
};
#endif

// Action lookup table to encode all of game state transitions so that the main game loop has as
// little branching as possible, to minimise divergence across CUDA threads playing independent
// games in parallel. The key state representation decision which enables this is that the pile is
// treated as a pseudo player, so the core game loop is always has one player playing their next
// card by adding it to another player's stack. This either the real players taking turns as
// required to add a card to the discard pile, or the discard pile "playing" all of its cards
// one-by-one (in separate turns) to the trick winner. To avoid counting the discard pile's turns,
// we make the turn and trick increments for each game cycle part of the action lookup information.

DEVICE_ONLY
//CONSTANT
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
            { discard_pile, discard_pile, pickup_by_1, 1, 0 },
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
            { discard_pile, discard_pile, pickup_by_0, 1, 0 },
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
            { player_0, player_0, discard, 0, 1 }
        },
        // state = pickup_by_1
        {
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, discard_pile, pickup_by_1, 0, 0 },
            { player_1, player_1, discard, 0, 1 }
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
    friend class StandardDeck;

    StackOfCards() = default;
    StackOfCards(StandardDeck& d);

    DEVICE_CALLABLE
    bool operator==(const StackOfCards& other) {
        if (num_cards() != other.num_cards())
            return false;
        for (unsigned i = 0; i < num_cards(); i++)
            if (cards[(i+first) & stack_mask] != other.cards[(i+other.first) & stack_mask])
                return false;
        return true;
    }

    DEVICE_CALLABLE
    bool operator!=(const StackOfCards& other) { return !(this->operator==(other)); }

    DEVICE_CALLABLE
    bool operator[](const int i) { return cards[(i+first) & stack_mask]; }

    DEVICE_CALLABLE
    void reset() {
        insert = first = 0;
    }

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

    std::string to_string() const {
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
    unsigned num_cards() const {
        return insert - first;
    }

    DEVICE_CALLABLE
    bool empty() const {
        return insert == first;
    }

    DEVICE_CALLABLE
    bool not_empty() const {
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


// Minimal representation of a shuffled deck.
class StandardDeck {
private:
    Card cards[52];

public:
    friend class StackOfCards;

    StandardDeck() = default;

    DEVICE_CALLABLE
    unsigned num_cards() const {
        return 52;
    }

    DEVICE_CALLABLE
    bool operator==(const StandardDeck& other) {
        if (num_cards() != other.num_cards())
            return false;
        for (unsigned i = 0; i < num_cards(); i++)
            if (cards[i] != other.cards[i])
                return false;
        return true;
    }

    DEVICE_CALLABLE
    bool operator!=(const StandardDeck& other) { return !(this->operator==(other)); }

    // Initialize with the full standard 52-card deck, in a canonical sort order.
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
    }

    std::string to_string() const {
        const char syms[] = {'-', 'J', 'Q', 'K', 'A'};
        char tmp[53];
        char *t = tmp;
        for (unsigned i = 0; i < sizeof(cards); i++) {
            auto card = cards[i];
            assert(card <= ace);
            *t++ = syms[card];
        }
        *t = 0;
        return std::string(tmp);
    }

    DEVICE_CALLABLE
    void swap(unsigned i, unsigned j) {
        auto ival = cards[i];
        auto jval = cards[j];
        cards[i] = jval;
        cards[j] = ival;
    }

    // Initialize the two hands from a starting deck
    DEVICE_CALLABLE
    void set_hands(StackOfCards& a, StackOfCards& b) {
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

StackOfCards::StackOfCards(StandardDeck& d) {
    first = 0;
    insert = 52;
    std::memcpy(cards, d.cards, sizeof(cards));
}


// Play out a deal using finite state machine approach, to provide better concurrency in CUDA warps.
DEVICE_CALLABLE
void play_fsm(StackOfCards& deal, unsigned& turns, unsigned& tricks) {
    turns = 0;
    tricks = 0;
    StackOfCards hands[NUM_PLAYERS];
    deal.set_hands(hands[player_0], hands[player_1]);
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
#ifndef __CUDACC__
#ifdef EXTRA_VERBOSE
        if (verbose && next_action.count_turn) {
            std::string a = hands[0].to_string();
            std::string b = hands[1].to_string();
            std::string p = hands[discard_pile].to_string();
            printf("FSM - Turn %d: %s/%s/%s\n", turns, a.data(), b.data(), p.data());
        }
#endif
        if (verbose && next_action.count_trick) {
            std::string a = hands[0].to_string();
            std::string b = hands[1].to_string();
            printf("FSM - Trick %d: %s/%s\n", tricks, a.data(), b.data());
        }
#endif
        if (next_action.count_trick && (hands[player_0].empty() || hands[player_1].empty())) {
            // Don't start another trick if the losing player is now out of cards.
            game_state = game_over;
        }
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

#ifdef USE_CUDA

// Class for efficient bulk playing of many deals.

// The first approach used a searcher per CUDA thread, matching the natural approach for CPU
// threads, but this meant CUDA threads quickly diverged, since games are different lengths and we
// shuffled just one deck between games on each thread.

// The second approach was designed to allow the CUDA threads to stay converged nearly all the time,
// by avoiding divergence at the end of a game. Instead, game end just requires picking the next
// unplayed deal from a large backlog and resetting relevant game state. The final state of each
// deal is recorded on the backlog for later analysis, so determination of the best deal does not
// slow down game play. (That is done as another bulk pass at the end.) This worked by sharing one
// searcher across the whole grid of blocks, containing per thread state for RNG and best deal
// tracking. There was one backlog of deals for the whole grid, which meant we were bottlenecked on
// global memory access to generate the deals and to play them.

// Latest approach is to have a searcher per thread block, rather than one for the whole grid of
// blocks, so we can rely almost exclusively on __shared__ memory which is block scoped and very
// fast. Each block (ie. searcher instance) therefore has its own small backlog of deals to play in
// shared memory, which is generated then played in short fast cycles, but using the same technique
// as before to ensure the threads for that block stay mostly converged. We only want just enough
// blocks to fill all of the SMs, and enough threads per block to fully use each SM's warp dispatch
// capacity (which is typically 2 warps of 32 threads each on eg. Turing architecture GPUs.)

#ifdef COMPACT_DECK
#define DEAL_CLASS StandardDeck
#else
#define DEAL_CLASS StackOfCards
#endif

#ifndef BLOCKS
#define BLOCKS 16
#define THREADS_PER_BLOCK 128
#endif
#define NUM_THREADS (BLOCKS * THREADS_PER_BLOCK)

#ifdef USE_SHARED_MEM_FOR_DEALS
// Assume 48KB of each SM's shared mem can be used
#define NUM_DEALS ((48*1024)/sizeof(DEAL_CLASS))
#define NUM_BATCHES 128
#else
#define NUM_DEALS (1024*1024/BLOCKS)
#define NUM_BATCHES 1
//#define USE_SHARED_TABLE 1
#endif

class BestDealSearcher : public Managed {
public:
    unsigned next_deal_to_play = 0; // put this first to guarantee alignment for atomicAdd
    // Best deals seen by this searcher (across all batches so far)
    unsigned best_turns = 0;
    unsigned best_tricks = 0;
    DEAL_CLASS best_turns_deal;
    DEAL_CLASS best_tricks_deal;
    // Track best deals seen so far by each thread (to avoid synchronization while playing).
    struct {
        unsigned best_turns;
        unsigned best_tricks;
        unsigned best_turns_deal;
        unsigned best_tricks_deal;
    } by_thread[THREADS_PER_BLOCK];
    // Use kernel version of curand in multi-threaded way
    curandState rng[THREADS_PER_BLOCK];
    // Keep the last deal from the previous search run to use as the starting point for the next run
    DEAL_CLASS starting_deck[THREADS_PER_BLOCK];
#ifndef USE_SHARED_MEM_FOR_DEALS
    // Batch of deals to play; this is not mutated, so we can read back the best deal(s) at the end.
    DEAL_CLASS deals[NUM_DEALS];
#endif

    BestDealSearcher() = default;

    DEVICE_ONLY
    void init(unsigned long long seed) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // Use multiple CUDA threads as recommended to parallelize the shuffling. We need to make
        // RNGs independent for every thread across the whole grid, but note we are only working
        // with this block's threads here.
        curand_init(seed, tid, 0, &rng[threadIdx.x]);
        // Start with a fixed per-suit descending order the very first time, for reproducibilty.
        starting_deck[threadIdx.x].set_full_deck();
    }

    DEVICE_ONLY
    void generate_batch(DEAL_CLASS deals[]) {
        unsigned int tid = threadIdx.x;
        // Work on a local copy of the rng state for speed
        curandState my_rng = rng[tid];
        // Continue shuffling from deals in the previous batch
        auto deck = deals[tid];
        auto num_cards = deck.num_cards();
        for (unsigned deal = tid; deal < NUM_DEALS; deal += THREADS_PER_BLOCK) {
            for (unsigned i = 0; i < num_cards-1; i++) {
                // Ignore the slight bias from using simple modulus on random unsigned
                unsigned offset = curand(&my_rng) % (num_cards-i);
                unsigned j = i + offset;
                deck.swap(i, j);
            }
            deals[deal] = deck;
        }
        rng[tid] = my_rng;
    }

    // Many CUDA threads will execute this in parallel. The aim is to keep threads as converged as
    // possible and to avoid synchronization, just using an atomic increment operation on
    // `next_deal_to_play` to pick the next game, so it doesn't matter how long games take to play.
    // It isn't clear whether it would be better to minimise the overhead of switching to a new
    // game, by moving game state into each deal so we can switch game by just changing `this_deal`.
    DEVICE_ONLY
    void play_batch(DEAL_CLASS deals[]) {
#ifdef USE_SHARED_TABLE
        // Keep action table in fast shared memory, rather than relying on caching.
        __shared__ NextAction shared_action_table[NUM_PLAYERS][NUM_STATES][NUM_CARDS];
        if (threadIdx.x == 0) {
            for (unsigned i=0; i < sizeof(action_table)/sizeof(unsigned); i++)
                ((unsigned *)shared_action_table)[i] = ((unsigned *)action_table)[i];
        }
        __syncthreads();
        #define ACTION_TABLE shared_action_table
#else
        #define ACTION_TABLE action_table
#endif
        unsigned int tid = threadIdx.x;
        unsigned turns = 0;
        unsigned tricks = 0;
        unsigned this_deal = 0;
        GameState game_state = game_over;
        Player player;
        Card next_card;
        StackOfCards hands[NUM_PLAYERS];
        by_thread[tid].best_turns = best_turns;
        by_thread[tid].best_tricks = best_tricks;

        while (true) {
            if (game_state == game_over) {
                if (turns > by_thread[tid].best_turns) {
                    by_thread[tid].best_turns = turns;
                    by_thread[tid].best_turns_deal = this_deal;
                }
                if (tricks > by_thread[tid].best_tricks) {
                    by_thread[tid].best_tricks = tricks;
                    by_thread[tid].best_tricks_deal = this_deal;
                }
                // pick up the next game to play, if there are any left in this batch
                this_deal = atomicAdd(&next_deal_to_play, 1);
                if (this_deal >= NUM_DEALS)
                    break;
                deals[this_deal].set_hands(hands[player_0], hands[player_1]);
                hands[discard_pile].reset();
                turns = 0;
                tricks = 0;
                player = player_0;
                game_state = discard;
            } else {
                // This slightly reduces throughput, though was suggested by Nsight Compute.
                //__syncwarp(0xffffffff);
                if (hands[player].not_empty())
                    next_card = hands[player].pop();
                else
                    next_card = no_card;
                NextAction &next_action = ACTION_TABLE[player][game_state][next_card];
                assert(next_action.next_state != logic_error);
                if (next_card != no_card)
                    hands[next_action.destination].append(next_card);
                player = next_action.next_player;
                game_state = next_action.next_state;
                turns += next_action.count_turn;
                tricks += next_action.count_trick;
                if (next_action.count_trick && (hands[player_0].empty() || hands[player_1].empty())) {
                    // Don't start another trick if the losing player is now out of cards.
                    game_state = game_over;
                }
            }
        }
    }

    DEVICE_ONLY
    void update_best_deals(DEAL_CLASS deals[]) {
        if (threadIdx.x != 0)
            return;
        for (unsigned tid = 0; tid < THREADS_PER_BLOCK; tid++) {
            if (by_thread[tid].best_turns > best_turns) {
                best_turns = by_thread[tid].best_turns;
                best_turns_deal = deals[by_thread[tid].best_turns_deal];
            }
            if (by_thread[tid].best_tricks > best_tricks) {
                best_tricks = by_thread[tid].best_tricks;
                best_tricks_deal = deals[by_thread[tid].best_tricks_deal];
            }
        }
    }
};


// ------------------------------------------------------------------------------------------
// Controller thread to report progress as potentially parallel search is in progress. One host
// thread passes work to a cuda stream to do a decent number of iterations, then processes the
// results. If the cuda stream is used as a filter to identify deals that don't complete within a
// threshold number of turns, then the host thread could work on playing each unfinished deal to get
// the final result, after dispatching the cuda stream to work on another set of games.

__global__
void cuda_init_searchers(BestDealSearcher *searchers, unsigned long long seed) {
    searchers[blockIdx.x].init(seed);
}

__global__
void cuda_run_searchers(BestDealSearcher *searchers) {
#ifdef USE_SHARED_MEM_FOR_DEALS
    __shared__ DEAL_CLASS deals[NUM_DEALS];
#define DEALS deals
#else
#define DEALS (searchers[blockIdx.x].deals)
#endif
    // Continue from where the previous search run ended
    DEALS[threadIdx.x] = searchers[blockIdx.x].starting_deck[threadIdx.x];
    for (unsigned batch = 0; batch < NUM_BATCHES; batch++) {
        searchers[blockIdx.x].generate_batch(DEALS);
        __syncthreads();
        searchers[blockIdx.x].play_batch(DEALS);
        __syncthreads();
        searchers[blockIdx.x].update_best_deals(DEALS);
        __syncthreads();
    }
    // Update starting point for the next search run
    searchers[blockIdx.x].starting_deck[threadIdx.x] = DEALS[threadIdx.x];
#undef DEALS
}

bool progress_printed = false;
unsigned global_best_turns = 0;
unsigned global_best_tricks = 0;
unsigned long long global_deals_tested = 0;

void run_search(unsigned long long seed) {
    BestDealSearcher *searchers = new BestDealSearcher[BLOCKS];

    cuda_init_searchers<<<BLOCKS, THREADS_PER_BLOCK>>>(searchers, seed);
    cudaDeviceSynchronize();
    clock_t start_time = clock();
    clock_t last_update_time = start_time;
    unsigned long long last_deals_tested = 0;

    while (true) {
        for (unsigned b = 0; b < BLOCKS; b++)
            searchers[b].next_deal_to_play = 0;
        cuda_run_searchers<<<BLOCKS, THREADS_PER_BLOCK>>>(searchers);
        cudaDeviceSynchronize();
        for (unsigned b = 0; b < BLOCKS; b++) {
            global_deals_tested += NUM_DEALS * NUM_BATCHES;
        }

        // Report progress and best deals so far, but at most once a second
        auto now = clock();
        double secs_since_last_report = (double)(now - last_update_time) / CLOCKS_PER_SEC;
        if (secs_since_last_report >= 1) {
            last_update_time = now;

            double secs_since_start = (double)(now - start_time) / CLOCKS_PER_SEC;
            printf("\r%g seconds, %llu deals tested (%g per second since start) (%g in last second)    ",
                   secs_since_start, global_deals_tested, global_deals_tested / secs_since_start,
                   (global_deals_tested - last_deals_tested) / secs_since_last_report);
            last_deals_tested = global_deals_tested;
            progress_printed = true;
            for (unsigned b = 0; b < BLOCKS; b++) {
                auto searcher = &searchers[b];
                if (searcher->best_turns > global_best_turns || searcher->best_tricks > global_best_tricks) {
                    if (progress_printed) {
                        printf("\n");
                        progress_printed = false;
                    }
                    if (searcher->best_turns > global_best_turns) {
                        global_best_turns = searcher->best_turns;
                        StackOfCards deck(searcher->best_turns_deal);
                        unsigned turns, tricks;
                        play(deck, turns, tricks);
                        printf("%s: %d turns, %d tricks\n", deck.to_string().data(), turns, tricks);
                        if (turns != global_best_turns) {
                            unsigned fsm_turns, fsm_tricks;
                            play_fsm(deck, fsm_turns, fsm_tricks);
                            printf("GPU got %u turns, CPU got %u, fsm got %u\n", global_best_turns, turns, fsm_turns);
                        }
                    }
                    if (searcher->best_tricks > global_best_tricks) {
                        global_best_tricks = searcher->best_tricks;
                        if (searcher->best_tricks_deal != searcher->best_turns_deal) {
                            StackOfCards deck(searcher->best_tricks_deal);
                            unsigned turns, tricks;
                            play(deck, turns, tricks);
                            printf("%s: %d turns, %d tricks\n", deck.to_string().data(), turns, tricks);
                            if (tricks != global_best_tricks) {
                                unsigned fsm_turns, fsm_tricks;
                                play_fsm(deck, fsm_turns, fsm_tricks);
                                printf("GPU got %u tricks, CPU got %u, fsm got %u\n", global_best_tricks, tricks, fsm_tricks);
                            }
                        }
                    }
                }
            }
        }
    }
}

#else // !USE_CUDA

void run_search(unsigned long long seed) {}

#endif // USE_CUDA

int main(int argc, char **argv) {
#ifdef USE_CUDA
    printf("%d/%d blocks/threads == %d searchers\n", BLOCKS, THREADS_PER_BLOCK, BLOCKS * THREADS_PER_BLOCK);
    printf("%u deals * %u batches * %u blocks == %u deals per search\n",
           (unsigned)NUM_DEALS, (unsigned)NUM_BATCHES, (unsigned)BLOCKS,
           (unsigned)(NUM_DEALS * NUM_BATCHES * BLOCKS));
    printf("sizeof(BestDealSearcher) is %zd bytes\n", sizeof(BestDealSearcher));
    printf("sizeof(StackOfCards) is %zd bytes\n", sizeof(StackOfCards));
    printf("sizeof(StandardDeck) is %zd bytes\n", sizeof(StandardDeck));
    printf("sizeof(curandState) is %zd bytes\n", sizeof(curandState));
    printf("sizeof(action_table) is %zd bytes\n", sizeof(action_table));
#endif

    if (argc == 1 || (argc == 3 && !strcmp(argv[1], "--seed"))) {
        // GPU search mode from fixed seed.
        int seed = 0;
        if (argc == 3 && !strcmp(argv[1], "--seed")) {
            seed = atoi(argv[2]);
        }
        run_search(seed);

    } else if (argc == 3 && !strcmp(argv[1], "test-fsm")) {
        // Play number of iterations to check that fsm and original version agree.
        int seed = 0;
        int iterations = atoi(argv[2]);
        std::mt19937_64 rng;
        // If seed is 0, leave rng in default initial state for reproducibilty
        if (seed != 0) {
            std::default_random_engine seeder(seed);
            std::seed_seq mt_seed{seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), };
            std::mt19937_64 new_rng(mt_seed);
            rng = new_rng;
        }

        StackOfCards deal;
        deal.set_full_deck();
        auto num_cards = deal.num_cards();
        bool keep_going = true;
        int deal_count = 0;
        while (keep_going) {
            for (unsigned i=0; i<num_cards-1; i++) {
                std::uniform_int_distribution<unsigned> u(0, num_cards-i-1);
                unsigned offset = u(rng);
                unsigned j = i + offset;
                if (deal.test_and_swap(i, j)) {
                    ++deal_count;
                    printf("\r[%4d] %s: ", deal_count, deal.to_string().data());
                    unsigned turns, tricks;
                    play(deal, turns, tricks);
                    unsigned fsm_turns, fsm_tricks;
                    play_fsm(deal, fsm_turns, fsm_tricks);
                    if (turns != fsm_turns || tricks != fsm_tricks) {
                        printf("FSM got %u/%u turns/tricks, CPU got %u/%u\n", fsm_turns, fsm_tricks, turns, tricks);
                    }
                    if (--iterations == 0) {
                        keep_going = false;
                        break;
                    }
                }
            }
        }

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
            //verbose = false;
            printf("Testing original play...\n");
        }
        play(deal, turns, tricks);
        printf("There were %d turns\n", turns);
        printf("There were %d tricks\n", tricks);
    }
    return 0;
}
