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
10 seconds, 56623104 deals tested (5.66231e+06 per second)
--J----J------KQ--K---A--A--AQ-Q--------J-KA---J--KQ: 4901 turns, 696 tricks
270 seconds, 1264582656 deals tested (4.68364e+06 per second)
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
