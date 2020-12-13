# Cafe: Predicting Physical Attractiveness with Deep Learning-Based Systems

## Abstract
The modern dating scene has been transformed by digital technologies. With the rise of social media apps such as Tinder and Bumble, finding a potential partner has never been so convenient. Can the process of finding a match be further digitized with deep learning techniques? This study explores the viability of using convolutional neural networks to create personalized matchmaking algorithms. Using transfer learning techniques, five architectures were appended to VGGNet, and trained on photos scraped from Google Images. Most networks were able to achieve an accuracy between 74% and 76%, competing with even the best wingman. This study also investigates how such models would perform on multiple images of a single person, which is intended to represent a dating profile. Most architectures exhibited reliable results, consistently giving the same label to four out of five images of a specific individual. Lastly, this paper explores the explainability of such models.

## Related Works
With the number of online singles growing, a person can get lost viewing profiles for many hours, never finding a single match. Waiting for your profile to be shown to others (and liked) can take time as well. On Tinder, a user can even be shown profiles they have already disliked as the pool of unviewed profiles dries up [27]. It has also been claimed that Tinder displays users that have already swiped left on you, in which case swiping on those profiles are meaningless. By increasing the volume and availability of prospective partners, these apps have also added an unintended inefficiency to this process. These undesirable app features, however, may be aimed more at maintaining a user base and following a business model than at efficient matchmaking.
To reduce wasted time, we propose using Deep Learning and transfer learning methods to automatically classify profiles for a given user. By training a personal algorithm, all matching would be done instantly (or in the required computational time to predict all profiles in the user-base) upon signing up for the service. Similar business models have been seen in social media apps such as Facebook, Instagram, and Twitter. Most recently, TikTok has exploited a prediction-based AI recommendation system for content feeds to foster substantial user-engagement rates. This is a large reason the average user spends 52 minutes per day on the app and the value of their parent company, ByteDance, grew to $75 million in November 2018 [7]. 
This study aims to provide a proof-of-concept for the idea and to define a model architecture that can be used generally to solve this task. Specifically, the model architecture was trained to predict physical attractiveness of a woman’s face according to the author’s personal preferences.