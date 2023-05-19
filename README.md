#  Introduction

A learning exercise in speeding up a computationally intensive parallelizable C++ program, by
leveraging CUDA GPUs.

In this case, the computationally intensive algorithm is to continuously play games of Beggar My
Neighbour, looking for the longest game (as measured either in player turns, or tricks played). It
is not known whether there is a game that will never end, though empirically it seems unlikely, and
the programs here all assume every game will finish in a finite number of moves.

This is essentially a toy algorithm, but it has some notable features which are relevant to the
challenge of effectively leveraging a GPU.

1. Game play is completely deterministic, so the outcome is defined only by the initial sorting of
   the deck. The problem is therefore embarrassingly parallel, since the large dimension is the
   number of distinct deals, which can in principle all be explored in parallel.   
2. The algorithm to play one game tracks game state by using a combination of a few simple state
   variables and branching logic. This is easy to write and verify, and also reasonably efficient to
   run on a single CPU core.
3. The core algorithm (to play one game) is very compact in terms of code size and data size. The
   surrounding algorithm to play many games while tracking the best ones so far is also naturally
   compact.

#  Steps In Improving Speed

## Starting Point

The original C++ program is `playbmn_cpu.cpp`. Using the Visual Studio C++ command line build tools
for Windows x64, it can be compiled and run like so:

```
C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
cl /Zi /EHsc /DNDEBUG /O2 playbmn_cpu.cpp
playbmn_cpu [-t N]
```

On my laptop with a Core i7-9750H, using a single thread, I get a sustained deal rate of about
614,000 deals per second. Using 6 threads (to match the physical core count) I get about 2.3M deals
per second. At 12 threads (to match the logical core count) I get close to 2.9M, which as expected
is as fast as it will go.


## GPU Baseline

The latest CPU version so far is `playbmn.cpp`. It performs slightly better than the original
version when using all logical cores, due to a tiny data structure tweak. It can also be compiled
with `nvcc` to run as a straight port to CUDA, basically just substituting CUDA threads for CPU
threads. I've chosen a CUDA blocksize and thread count that seems to be optimal for my laptop (32
for each).

```
copy playbmn.cpp playbmn_gpu.cu
nvcc -DUSE_CUDA -DNDEBUG -lcurand -o playbmn_gpu.exe playbmn_gpu.cu
playbmn_gpu
```

On my laptop the GPU baseline manages about **2.3M deals per second**, using a GeForce GTX 1650 which
has 896 CUDA cores. The interesting comparison is with the CPU version using all logical cores (12
in my case), processing about **3M deals per second**.

For reference, compute utilization on the GPU is 95%, while mem utilization is 0% (reflecting
the unusual fact that we barely use any GPU memory, since we are iteratively shuffling and playing
one deck per thread, and game play state is all in registers).


## State Machine with Incremental Shuffling

In an effort to minimise CUDA thread divergence within each warp of threads, I replaced the core
game play algorithm (which relies on nested branching logic) with a state machine structure. That
uses a central lookup table to map current state and latest input to the next state values. The idea
is that all the CUDA threads will normally execute the lookup steps in unison, since the game state
is now tracked purely in data and not also using the code position within the nested branching
logic. The net effect should be more thread concurrency (since when divergence happens, threads
execute potentially one at a time until they reach the same point in the code again).

The search logic is unchanged; after each game finishes, the deck for that thread has a pair of
cards swapped, to play the next game.

The state machine version is found in `playbmn_fsm.cpp`, which can be compiled for CPU or GPU.

```
copy playbmn_fsm.cpp playbmn_fsm_gpu.cu
nvcc -DUSE_CUDA -DNDEBUG -lcurand -o playbmn_fsm_gpu.exe playbmn_fsm_gpu.cu
playbmn_fsm_gpu
```

Sadly, this results in a big step backwards in performance, from 2.3M to around 1.4 - 1.6M deals per
second on GPU. (There is a similar drop in performance on CPU; branch prediction and speculative
execution on CPU handles nested branching logic very well.)

Compute utilization on the GPU is now around 80%, mem utilization is about 30%, since we are heavily
accessing the lookup table in GPU memory but not able to use the system's full memory bandwidth.


## State Machine with Large Backlog of Deals

I realised thread divergence still happened because games are different lengths, and end of game was
forcing an implicit thread synchronization (since the "play" function exits back to the search loop
when each game finishes). To avoid this, I now precreate a large backlog of deals and have each CUDA
thread pick the next game from the list whenever it finishes its current game. A single low-cost
atomic operation is enough to do this without a full thread sync.

At this point, the code is now quite CUDA specific, so I've forked it into `playbmn_cuda_fsm.cpp`
which only compiles for GPU.

```
copy playbmn_cuda_fsm.cpp playbmn_cuda_fsm.cu
nvcc -DOVERRIDES -DBLOCKS=32 -DTHREADS_PER_BLOCK=32 -DUSE_CUDA -DNDEBUG -lcurand -o playbmn_cuda_fsm.exe playbmn_cuda_fsm.cu
playbmn_cuda_fsm
```

This improves speed compared to the initial State Machine version, but only to about 2.8M deals per
second at best (degrading to 2.2M or so under thermal throttling), so roughly the same as the
baseline version. Surely something is still not being done right? Time to look at what the Insight
GPU runtime performance analysis tool can tell us!

Compute utilization on the GPU is now around 98%, mem utilization is about 15% (also indicating
memory use is not efficient yet). Changing blocks/threads to 16/128 has similar results, but
improves mem utilization to 25%.


## State Machine with Lookup Table in Shared Memory

Based on information from the Insight GPU instruction runtime analysis tool, that memory access was
a big bottleneck and compute resources are barely being used, I tried placing the state machine
lookup table in SM Shared Memory, since it is accessed in the inner game loop. For reference, shared
memory is a very small space (48KB) of very fast SRAM, where access speed is virtually the same as
register access speed. The table has to be copied into SRAM for each block of threads though.

```
copy playbmn_cuda_fsm.cpp playbmn_cuda_fsm.cu
nvcc -DOVERRIDES -DUSE_SHARED_TABLE -DUSE_CUDA -DNDEBUG -lcurand -o playbmn_cuda_fsm.exe playbmn_cuda_fsm.cu
playbmn_cuda_fsm
```

Basically this results in no change. I believe it is because in the previous version the lookup
table fits in L1 cache, which is also in SRAM alongside shared memory. Actually this version is
slightly slower, presumably from the small overhead of copying the lookup table into shared memory
for each thread block.

Compute utilization on the GPU is still around 98%, mem utilization about 25% (still indicating
memory use is not that efficient yet).


## State Machine with Small Backlogs in Shared Memory

Since memory access speed is still the primary bottleneck, it must be related to reading from the
large backlog in main GPU memory (on the order of 20x slower than SRAM), even though we aren't doing
that in the innermost game play loop. So let's just use a small backlog in SRAM and not touch main
GPU memory very much. (We leave the lookup table in main GPU memory, relying on L1 cache to handle
it and leaving more shared memory for the backlogs.) This has a minor drawback that we don't have a
single backlog shared perfectly across all threads, but instead a small backlog for each block of
threads -- since game lengths can vary quite a lot, we will have individual threads run out of games
to play more often.

```
copy playbmn_cuda_fsm.cpp playbmn_cuda_fsm.cu
nvcc -DOVERRIDES -DUSE_SHARED_MEM_FOR_DEALS -DUSE_CUDA -DNDEBUG -lcurand -o playbmn_cuda_fsm.exe playbmn_cuda_fsm.cu
playbmn_cuda_fsm
```

However, even with the minor drawback of losing a shared backlog, this finally gives us a big gain.
Performance pulls ahead of CPU baseline by an order of magnitude, achieving roughly 40M deals per
second (which drops off somewhat when thermal throttling kicks in, to about 36M on average). Can we
do even better though? Insight tools are still saying compute resources aren't being used very
effectively.

Compute utilization on the GPU has dropped quite a lot to 70%, and mem utilization is down to about
15%. On the face of it, this looks bad, yet the game playing speed is obviously the best so far. So
it seems there could still be room for improvement, if we can make progress on the poor memory
utilization.


## State Machine with Small Backlogs in Shared Memory, uint8 enum type

Since everything important is already in shared memory, the next step is to make those data
structures more compact. The final optimization I made was to shrink them by changing the base type
for C++ enums from `int` (32-bits) to `uint8` (8-bits), packing the lookup table cells into fewer
bits, and storing deals in the backlogs in a smaller data structure (not the larger one that is
optimized for game playing). The benefit is two-fold: more deals can be precreated in a batch
operation when the backlog needs to be refilled, and less memory (SRAM) bandwidth is needed per
cycle of game play.

```
copy playbmn_cuda_fsm.cpp playbmn_cuda_fsm.cu
nvcc  -DUSE_CUDA -DNDEBUG -lcurand -o playbmn_cuda_fsm.exe playbmn_cuda_fsm.cu
playbmn_cuda_fsm
```

The cumulative effect is finally what I was hoping for! Sustained speed is around 100M deals per
second, which is 43x what the initial port to GPU achieved, and 33x the CPU baseline we started
with.

Compute utilization on the GPU is now back up, to 93%, mem utilization is actually at 1%. I think
this means mem utilization reflects bandwidth to main GPU memory, which in our case (rather
unusually) can be almost completely avoided. This would seem to be about as good as we can hope to
get.


---------------


#  Log of Results

Rough history of improvements, from initial trivial port to CUDA, to latest refinement.

All speeds are in millions of deals played per second, on laptop with i7-9750H and GeForce GTX 1650.
For fair comparisons, compile with NDEBUG defined in order to compile out asserts.

CPU baseline (playbmn.cpp):
- 1 thread:    0.56   (CPU running at 4.0GHz)
- 2 threads:   1.1   (4.0GHz)
- 4 threads:   1.7    (3.3GHz)
- 8 threads:   2.5    (2.9GHz)
- 12 threads:  3.0    (2.8GHz)
- 16 threads:  3.0    (2.8GHz)

GPU baseline (playbmn.cpp):
- 32 blocks of 32 threads:    2.3

CPU lookup table (playbmn_fsm.cpp):
- 1 thread:    0.27   (CPU running at 4.0GHz)
- 2 threads:   0.52   (4.0GHz)
- 4 threads:   1.0    (3.7GHz)
- 8 threads:   1.4    (2.7GHz)
- 12 threads:  1.9    (2.7GHz)
- 16 threads:  1.9    (2.7GHz)

GPU lookup table (playbmn_fsm.cpp):
- 128 blocks of 32 threads

--> 1.4

GPU lookup table with continual play from "global" backlog (playbmn_cuda_fsm.cpp):
- without USE_SHARED_TABLE
- without USE_SHARED_MEM_FOR_DEALS
- default enum and action_table size
- DEAL_CLASS is StackOfCards
- 32 blocks of 32 threads

--> 2.5   (P0, 75C, 98% sm util)
--> 2.3   (P3, 74C, 97% sm util)

GPU "shared" lookup table with continual play from "global" backlog (playbmn_cuda_fsm.cpp):
- with USE_SHARED_TABLE
- minimum enum and action_table size
- without USE_SHARED_MEM_FOR_DEALS
- DEAL_CLASS is StackOfCards
- 32 blocks of 32 threads

--> 3.5 ??

....

GPU "shared" lookup table with continual play from "global" backlog (playbmn_cuda_fsm.cpp):
- without USE_SHARED_TABLE
- minimum enum and action_table size
- with USE_SHARED_MEM_FOR_DEALS
- DEAL_CLASS is StandardDeck
- 16 blocks of 128 threads

--> ~100

_Old Runs (To be replaced with more precise info)_

2023-03-29: My best run by far (ie. longest game). This is also the fastest version of the program
so far, last config as described above, averaging 98m deals per second on a GeForce GTX 1650:

```
playbmn_cuda_fsm --seed 987654321
16/128 blocks/threads == 2048 searchers
945 deals * 128 batches * 16 blocks == 1935360 deals per search
sizeof(BestDealSearcher) is 14968 bytes
sizeof(StackOfCards) is 72 bytes
sizeof(StandardDeck) is 52 bytes
sizeof(curandState) is 48 bytes
sizeof(action_table) is 1008 bytes
1.013 seconds, 116121600 deals tested (1.14631e+08 per second since start) (1.14631e+08 in last second)
Q---J--J--J----A-------K-Q-----A-Q----KKA-A-KJ-Q----: 2508 turns, 340 tricks
-----A--Q------AA-K---Q-QA-QJK-K-J-JJ----------K----: 2498 turns, 344 tricks
J------J----J-QQ--K---A----JKQ----------KQ--A-K--A-A: 2641 turns, 365 tricks
3.024 seconds, 356106240 deals tested (1.1776e+08 per second since start) (1.18687e+08 in last second)
-KAJ---K--K-Q-K--------Q--QA--Q--------A--JJA------J: 2879 turns, 404 tricks
6.066 seconds, 704471040 deals tested (1.16134e+08 per second since start) (1.14518e+08 in last second)
-JK---A---A-J-A------A-K-----QQ---K-Q--Q-J-J--K-----: 3997 turns, 552 tricks
439.123 seconds, 43222394880 deals tested (9.84289e+07 per second since start) (9.87034e+07 in last second)
--Q--------J--A-------J----QKQK--A----K-JA-K-Q--JA--: 4005 turns, 566 tricks
488.624 seconds, 48107243520 deals tested (9.84545e+07 per second since start) (9.88593e+07 in last second)
K-----A-----QA---QQAK---J------QKJ-------K-J--A----J: 6005 turns, 839 tricks
```


Best run so far, making Card uint8_t, and packing the action_table entries into fewer bits.

```
C:\Users\AndrewInnes\home\progs\bmn>playbmn_cuda_fsm
32/32 blocks/threads == 1024 searchers
sizeof(BestDealSearcher) is 75538592 bytes
sizeof(StackOfCards) is 72 bytes
sizeof(action_table) is 1008 bytes
1 seconds, 6291456 deals tested (6.29146e+06 per second)
Q-----KQ---QA--J--J-A-A-J---QK-----K----JA------K---: 3316 turns, 457 tricks
2 seconds, 12582912 deals tested (6.29146e+06 per second)
-----J--A----Q-A-A-Q--JK-K-J----Q---Q-A-J-K------K--: 3556 turns, 490 tricks
3 seconds, 18874368 deals tested (6.29146e+06 per second)
-Q--K----Q-JAK-------J----AK--A---JA---KQ----Q-----J: 3737 turns, 516 tricks
4 seconds, 25165824 deals tested (6.29146e+06 per second)
-----A----K--A----KQ------K--JQJQA------A-Q------KJJ: 3741 turns, 513 tricks
9 seconds, 56623104 deals tested (6.29146e+06 per second)
--J----J------KQ--K---A--A--AQ-Q--------J-KA---J--KQ: 4901 turns, 696 tricks
3341 seconds, 15514730496 deals tested (4.64374e+06 per second)
Q-K-----JJQ------K----A-K-JA------------KJ-A-Q--Q-A-: 4983 turns, 698 tricks
5851 seconds, 27269267456 deals tested (4.66062e+06 per second)
--------KQ---A---QJKJ---Q---K-----JJ--AQ-AK---A-----: 5603 turns, 765 tricks
22124 seconds, 103898152960 deals tested (4.69617e+06 per second)
```


Previous best run so far, which is using a ` __shared__` copy of `action_table`.

Nsight Compute indicates it is memory bound, with low utilization of compute resources (around 10%
of SOL, vs. 60% for memory). For reference, using 8 CPU threads on laptop with 8 cores, the original
algorithm manages about 2.1e6 deals per second. Using 16 CPU threads, it runs at 2.2e6.

```
C:\Users\AndrewInnes\home\progs\bmn>playbmn_cuda_fsm
32/32 blocks/threads == 1024 searchers
sizeof(BestDealSearcher) is 276865568 bytes
sizeof(StackOfCards) is 264 bytes
sizeof(action_table) is 2016 bytes
1 seconds, 5242880 deals tested (5.24288e+06 per second)
Q-----KQ---QA--J--J-A-A-J---QK-----K----JA------K---: 3316 turns, 457 tricks
2 seconds, 10485760 deals tested (5.24288e+06 per second)
-----J--A----Q-A-A-Q--JK-K-J----Q---Q-A-J-K------K--: 3556 turns, 490 tricks
3 seconds, 15728640 deals tested (5.24288e+06 per second)
-Q--K----Q-JAK-------J----AK--A---JA---KQ----Q-----J: 3737 turns, 516 tricks
5 seconds, 26214400 deals tested (5.24288e+06 per second)
-----A----K--A----KQ------K--JQJQA------A-Q------KJJ: 3741 turns, 513 tricks
12 seconds, 57671680 deals tested (4.80597e+06 per second)
--J----J------KQ--K---A--A--AQ-Q--------J-KA---J--KQ: 4901 turns, 696 tricks
3858 seconds, 15517876224 deals tested (4.02226e+06 per second)
Q-K-----JJQ------K----A-K-JA------------KJ-A-Q--Q-A-: 4983 turns, 698 tricks
6782 seconds, 27266121728 deals tested (4.02037e+06 per second)
--------KQ---A---QJKJ---Q---K-----JJ--AQ-AK---A-----: 5603 turns, 765 tricks
28514 seconds, 114478284800 deals tested (4.01481e+06 per second)
```


Original best run:

```
C:\Users\AndrewInnes\home\progs\bmn>playbmn_cuda_fsm
32/32 blocks/threads == 1024 searchers
sizeof(BestDealSearcher) is 276890144 bytes
sizeof(StackOfCards) is 264 bytes
sizeof(action_table) is 2016 bytes
1 seconds, 4194304 deals tested (4.1943e+06 per second)
Q-----KQ---QA--J--J-A-A-J---QK-----K----JA------K---: 3316 turns, 457 tricks
4 seconds, 16777216 deals tested (4.1943e+06 per second)
-------J----Q-------A---J---AJ-K-Q--K---K-K--Q-QAJ-A: 3573 turns, 507 tricks
8 seconds, 27262976 deals tested (3.40787e+06 per second)
--AK--A--J--J----A-------JK-AQ----QK-----J--QQ----K-: 3648 turns, 499 tricks
28 seconds, 90177536 deals tested (3.22063e+06 per second)
-JJ--A------K-J-Q-Q-K--QA---A----A----------QKJ--K--: 3622 turns, 512 tricks
32 seconds, 102760448 deals tested (3.21126e+06 per second)
JQ----Q----K-----------A-K-JA--JQ--A--A--KQ-J-----K-: 3809 turns, 530 tricks
33 seconds, 105906176 deals tested (3.20928e+06 per second)
--J-Q-AK-K--K----A-----KQJ-------Q-J-A-J-----A--Q---: 3761 turns, 541 tricks
52 seconds, 165675008 deals tested (3.18606e+06 per second)
----A---Q----Q---JK-KQJJK---A----KQ----J--------A-A-: 3820 turns, 525 tricks
66 seconds, 209715200 deals tested (3.1775e+06 per second))
--AAJAKK-----J---AJ-Q-JQ---------K-------QKQ--------: 4340 turns, 599 tricks
91 seconds, 288358400 deals tested (3.16877e+06 per second)
-KAQQ-----J--J------K--JA--A----------K--K---A-Q--QJ: 4490 turns, 637 tricks
476 seconds, 1590689792 deals tested (3.34179e+06 per second)
----QA-----A----Q--Q-K-JAQ---------KJ--J------AKJK--: 4517 turns, 635 tricks
1641 seconds, 5618270208 deals tested (3.42369e+06 per second)
J-A---J-------QK-------JQQ--AK--Q-----KA--K------J-A: 4825 turns, 665 tricks
2643 seconds, 9073328128 deals tested (3.43297e+06 per second)
K------QK-Q---JA--Q--KQ--AJA---J-----A-----J-K------: 4876 turns, 692 tricks
8988 seconds, 31030509568 deals tested (3.45244e+06 per second)
Q-K-----JJQ------K----A-K-JA------------KJ-A-Q--Q-A-: 4983 turns, 698 tricks
15780 seconds, 54531194880 deals tested (3.45572e+06 per second)
--------KQ---A---QJKJ---Q---K-----JJ--AQ-AK---A-----: 5603 turns, 765 tricks
16745 seconds, 57864617984 deals tested (3.45564e+06 per second)
Q----J-K-A----J---------JK-----A-Q-A--QA----K-JK---Q: 5486 turns, 773 tricks
21748 seconds, 75084333056 deals tested (3.45247e+06 per second)
```
