When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don’t end up going through with a purchase during that web browsing session. It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer if the website believes the user isn’t planning to complete the purchase. How could a website determine a user’s purchasing intent? 

My task was to build a nearest-neighbor classifier to solve this problem. Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — my classifier will predict whether or not the user will make a purchase. To train the classifier, data was used from a shopping website from about 12,000 users sessions.

Accuracy was measured with two values: sensitivity (also known as the “true positive rate”) and specificity (also known as the “true negative rate”). The goal was to build a classifier that performs reasonably on both metrics.

After the program was run, the results were as follows:

- Sensitivity = 40.78%
- Specificity = 90.03%
