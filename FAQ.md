# Answers to Frequently Asked Research Questions

## Q1. Why there is no label that can definitively indicate similarity of interests?

The complete question is:
> Since it is mentioned in the introduction that item categories can be used to estimate whether search and recommendation behaviors exhibit similar interests, why not use this as a label for similarity of interests?

First, let us revisit the estimation method mentioned in the introduction. *``For each search behavior, if the categories of the items exist in the set of categories of the items interacted by this user in the past seven days, this search behavior is similar to recent recommendation behaviors, and otherwise dissimilar.''*

The approach mentioned in the text can only estimate the similarity of interests and cannot serve as a ground-truth label. This is because the category of an item can only be used as a rough criterion to approximate the user's interest category. An item may not necessarily belong to only one category, and it may have multiple hierarchical levels, with each level having multiple categories. Therefore, the accuracy of the category labels will affect the accuracy of the estimation results. For example, suppose a user's browsing history includes a basketball-related video, a football-related video, a badminton-related video, and then the user searches for a table tennis-related video. If we simply classify all videos as sports videos, then the search is related to recommendations. However, if we classify all videos according to different types of balls, then the search and recommendation become unrelated. Therefore, using category estimation can only serve as an estimation and cannot be used as ground-truth.

## Q2: Why use a user's entire search history and recommendation history to calculate similar and different interests?

The complete question is :
> Since a user may search for videos that are different from the recommended ones because they are dissatisfied with the recently recommended videos, or they may search for related videos only if they are satisfied with the recent videos, why not directly disentangle the user's interests by using search behaviors and recent recommendation behaviors?

Because a user's interests dynamically evolve, as illustrated in the example given in the previous answer, a user may not be interested in the videos they have recently watched, which leads them to search for videos related to badminton. From the current perspective, badminton is indeed a different interest from those recommended and searched for. However, as time goes on and the recommendation model learns about the user's interest in badminton, it will start recommending badminton-related videos to the user, and the user's browsing history on the recommendation service will gradually increase with badminton-related videos. At this point, badminton should be considered as a shared interest in both search and recommendation. Furthermore, considering that users often repeat their search behaviors, some search content that was a new interest during the first search may become a shared interest between search and recommendation after repeated searches.