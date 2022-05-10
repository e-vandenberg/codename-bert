# codename-bert
A Siamese BERT based network that plays Codenames

## What is Codenames?
Codenames - or at least this simplified version of it - is a board game in which there are a set of words on the table, some of which correspond to winning words, and other correspond to losing words. Only the **hint giver** is aware of which words are winning and losing, and they need to give a one word hint to the **guesser** to figure out which words will win them the game. 

In this implementation, a human is the hint giver, and the network is the guesser. In other words, the human player will look at the board and give the neural network a one word hint, which will then give it's best guess of which words you were trying to get it to guess.

For example, if the board looked like this:

![Screen Shot 2022-05-10 at 10 36 23 AM](https://user-images.githubusercontent.com/23105545/167654553-381b26b9-ad12-492f-9b3f-9a2016a5ee71.png)

Where the green words are winning words, the human player may want to provide a hint such as "butterfly" to ensure the guesser can find the winning words.

## How It Works
A siamese neural network passes two inputs to the same network, and then learns to classify the difference between those two outputs. It is a way to determine whether two words are similar or not, even if our classifier hasn't yet been trained on either of them. This technique is used in a number of real world applications, and is especially prominent in areas like facial recognition. 

![Screen Shot 2022-05-10 at 10 49 18 AM](https://user-images.githubusercontent.com/23105545/167657226-f22c018a-1352-4831-80cf-dfee44a48b12.png)

So in order to play Codenames, our network can run the input word alongside each board word through the siamese network and obtain a similarity score for each pair. From there, it can just pick the words with higher scores, and reject the ones with lower scores.
