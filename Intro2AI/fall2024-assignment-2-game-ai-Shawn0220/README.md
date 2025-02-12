# Assignment 2 - Game AI

## The Premise

After you sold your unicorn ChowRider, you used your massive windfall to travel the world and eat good food exclusively at Michelin-starred restaurants all over, eventually coming back stateside to settle down in a modest, 20-room mansion in Winnetka, right on the lake just a bit North of Evanston. One gloomy and gray morning, staring out your floor-to-ceiling living room window, you realized that for a bored billionaire who had everything, there was one thing you didn't actually have! (No, it's not the Continuum Transfunctioner. You have that too.) Your pristine, lush, green yard between your house and your private beach with the best cabana nobody has ever seen is missing one thing that you simply must have -- an 8x8 chess/checkers board with humongous game pieces! You know, the kind that Ron Weasley played that amazing game of chess on in *Harry Potter and the Sorcerer's* Stone. Three business days later, you had that too. All you need now is to invite a former classmate from your Intro to AI class at Northwestern to play checkers with them. You'd play chess, but while your talents are undeniable, playing chess isn't one of them, so you're going with checkers. But mischief has gotten the better of you. You want to put your recently gotten brain implant to the test, so you want an AI to help you to win by sending the best moves right to your head from an innocent little Mac tucked away upstairs in your study, communicating with your brain remotely. You can spend a buck or million on the best checkers AI they got out there, but remember, you're a bored billionaire, so you decided to write your own game AI. And you really miss the marvelous Intro to AI course you took with the most amazing professor who is so modest that he'd never consider himself amazing, so you decided to implement the beautiful adversarial search algorithm you learned about in that course -- Minimax with Alpha-Beta Pruning.

You call up your buddies at TestFriendForever (TFF) to ask them to generate some tests for you that you plan on using to test your implementation of Minimax with Alpha-Beta Pruning. They oblige and author some tests for you (and some secret ones they'll use to test your code to give it a gold star) and they also cut your work out for you based on a codebase they're supplying you with as a framework.

## The Task


In order to get the TFF gold star and impress (or anger) your former classmate, all you have to do is to implement the two (2) unimplemented functions in the file `ai.py` that are preceded by the string `"TO DO"` such that they pass the 21 tests at the bottom of the same file. The focus of your implementation is two-fold:

1. Compute certain metrics that are used by the `evaluate()` function. You'll do this in the `counts()` function. Inspecting `evaluate()` closely will tell you everything you need to know about the metrics that you need to compute inside `count()`.
2. Implement Minimax with Alpha-Beta Pruning in `minimax_alpha_beta()`, which is basically the brain of your game AI.

## The Details

TFF has provided you with some details they've urged you to keep in mind:

1. You must leave `evaluate()` untouched and only implement `counts()`, which will compute the various metrics.
2. One metric is the difference between how many moves you have available given a game board and how many your opponent does, and a hint from TFF is that when calculating the moves, you can keep track of any possible captures and possible king promotions along the way, which are two of the other metrics.
3. When calculating moves, you don't need to and must not consider multi-hop moves. While multi-hop moves are standard in checkers and will also be used by the game engine that will use your AI implementation, you should consider a move being one that is either to a diagonally adjacent square or to a square that's one diagonal hop away. As such, during a move, you can capture at most one piece. The tests assume this for counting available moves, so you must stick with it.
4. A `king hopeful`, something you'll see mentioned in the code, is an opportunity for a piece to become king if it makes a certain move. So, if a piece can make two moves and either move can make it a king, then the count of king hopefuls is 2.
   
## The Visualizations

As you'll see in the code, the `minimax_alpha_beta()` function already has a bit of implementation. It's there so the game engine works right off the bat and you can see the game board AND use your mouse to play a game of checkers! The function now chooses moves randomly, so if you beat the current "AI" in a game of checkers, you might want to hold off on the celebrations. Your task, as stated above, is to replace all the code in that function with your own implementation. Look at other parts of the codebase to identify which function you can use to visualize any game board at any point of gameplay. You can also ingest or output boards from and to text format, which you can also print to your terminal or output window.

## The Rules

1. You must implement **both (2)** unimplemented functions in `ai.py`.
2. You must not edit any file other than `ai.py`, unless you want to try out your own game board configurations, in which case you may add boards to `board_configs.py` starting from after board #28.
3. You may add to `ai.py` any number of helper functions you may need, but you must not modify any of the functions other than the two you are to implement.
4. You should feel invited to use Python modules as necessary, but you must not use functions or packages that implement Minimax with Alpha-Beta Pruning.
