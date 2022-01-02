#include <iostream>
#include <random>
#include <utility>
#include <cassert>

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
    StackOfCards() = default;

    // Initialize the stack with the full standard deck, in a canonical sort order.
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

    void swap(unsigned i, unsigned j) {
        assert(i < num_cards());
        assert(j < num_cards());
        std::swap(cards[(first+i) & stack_mask], cards[(first+j) & stack_mask]);
    }

    Card pop() {
        assert(num_cards() <= 52);
        assert(num_cards() > 0);
        return cards[(first++) & stack_mask];
    }

    void append(Card c) {
        assert(num_cards() < 52);
        cards[(insert++) & stack_mask] = c;
    }

    // Pick up a stack of cards (append here and leave the source stack empty)
    void pick_up(StackOfCards& s) {
        for (unsigned i = s.first; i < s.insert; i++) {
            append(s.cards[i & stack_mask]);
        }
        s.insert = s.first;
    }

    unsigned num_cards() {
        return insert - first;
    }

    bool not_empty() {
        return insert != first;
    }

    // Initialize the two hands from a starting deck
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
void play(StackOfCards& deal, unsigned& turns, unsigned& tricks)
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
        std::cout << "Starting hands: " << a << "/" << b << std::endl;
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
                std::cout << "Turn " << turns << ": " << a << "/" << b << "/" << p << std::endl;
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
            std::cout << "Trick " << tricks << ": " << a << "/" << b << "/" << p << std::endl;
        }
    }
}

unsigned best_turns = 0;
unsigned best_tricks = 0;
unsigned long long deals_tested = 0;

void track_best_deal(StackOfCards& deal) {
    unsigned turns, tricks;
    play(deal, turns, tricks);
    ++deals_tested;
    if (turns > best_turns || tricks > best_tricks) {
        std::cout << deal.to_string() << ": " << turns << " turns, " << tricks << " tricks" << std::endl;
        if (turns > best_turns)
            best_turns = turns;
        if (tricks > best_tricks)
            best_tricks = tricks;
    }
}

int main(int argc, char **argv) {
    if (argc == 2) {
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

        unsigned turns, tricks;
        play(deal, turns, tricks);
        std::cout << "There were " << turns << " turns" << std::endl;
        std::cout << "There were " << tricks << " tricks" << std::endl;

        return 0;
    }

    // Normal invocation, to conduct pseudo-random search for a deal that produces the longest game.
    // The basic approach is to start with a default sorted deck state, and then incrementally
    // shuffle this to generate deals which are different from ones we've already tested. We want
    // this to be very efficient, and as guaranteed as possible to not repeat previously seen deals,
    // or at least to not get stuck in a loop that won't explore new possible deals.
    //
    // It is well-known that a simple random shuffle (linear scan with pair swapping among the
    // remaining elements) will provide an unbiased random selection of a new permutation. So this
    // will be used here, except we will also test games at all of the intermediate swap points. A
    // similar approach could be used where a simple rotating position is used as the swap target,
    // with the other element chosen at random.

    StackOfCards deck;
    //std::default_random_engine seeder(1);
    //std::seed_seq mt_seed{seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), seeder(), };
    //std::mt19937_64 rng(mt_seed);
    std::mt19937_64 rng;

    // Start with a fixed per-suit descending order, for reproducibilty.
    deck.set_full_deck();
    auto num_cards = deck.num_cards();
    while (true) {
        for (unsigned i=0; i<50; i++) {
            std::uniform_int_distribution<unsigned> u(0, num_cards-i-1);
            unsigned j = i + u(rng);
            deck.swap(i, j);
            track_best_deal(deck);
        }
    }
}
