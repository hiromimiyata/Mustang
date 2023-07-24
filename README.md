# Mustang
## Project 4 Objective
To utilize and train a machine learning model to predict outcomes based a dataset.

### Goal
To create and optimize the best machine learning model to predict the winning horse.

### Data Collection
Used 2 datasets from this website on horse-racing data around the world in the year 2020.
https://www.kaggle.com/datasets/hwaitt/horse-racing

### Data Cleaning
Merged the two datasets and dropped unecessary columns and rows with missing information. Utilized MongoDB and SQL to input the dataset into a database to further analyze the data with different tools.

### Attempted Machine Learning Models
Tested multiple machine learning models including Neural Network, Decision Tree, Random Forest, K-Nearest, etc. The machine leaning models used in the presentation were the models with the best outcomes. The models mentioned were Neural Networks and the Decision Tree.

### Presentation Slides
![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2010.23.09.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.11.50.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2010.24.01.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2010.24.15.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.14.26.png)

 In our dataset, we identified the top 10 features that greatly impact race outcomes. Among them, "ncond," which represents the condition type, emerges as a critical factor. Conditions such as weather, track surface, and race category can significantly influence a horse's performance, making it a key determinant in predicting race results.
Following closely is the "race class" feature, which reflects the competitiveness and prestige of a race. Higher-class races attract top-quality horses, leading to intense competition and potentially higher winning chances for skilled racehorses.
Another influential factor is the "jockey name." This feature indicates the jockey's ability to communicate effectively with the horse, strategize during the race, and make crucial decisions that can greatly influence the horse's performance.
While "runners" and "age" appeared towards the end of our top 10 features list, they should not be underestimated in their importance. "Runners" represent the total number of horses competing in a race, and it can have a significant impact on the dynamics of the race. Larger fields can lead to more crowded and unpredictable races, making it crucial for horses to maneuver effectively to secure a favorable position. Similarly, "age" plays a critical role in determining a horse's performance. Younger horses, such as 2-year-olds, may have the advantage of speed and potential, while 3-year-olds might have gained valuable racing experience. As horses age beyond 3, their physical abilities may start to decline, affecting their racing performance.

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.02.53.png)

In this chart, we explored the relationship between the "isfav" (Is Favorite) feature and race outcomes. The dark blue areas represent races where the horse was the favorite ("isfav" = 1) the lighter blue areas represent races where the horse positioned or not. The orange areas indicates where the horse win or not. In some cases the favourite horse may have performed well and placed, in some instanced our favourite horse is not even positioned or win. So this finding shows the while being the favourite horse improves the probability of winning and placing, but its not guarentee of victory in every race. Horses that are not the favourite can still emerge victorious. Therefore, while "isfav" is an important feature in predicting race outcomes, it is not the sole determinant of a horse's success. Additional factors such as age, race conditions, jockey expertise, and the overall quality of the competition also play pivotal roles in shaping the race results.

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.03.46.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.04.05.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.04.15.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.04.35.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2021.15.05.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2021.15.18.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.05.13.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.05.50.png)

![Alt Text](https://github.com/hiromimiyata/Mustang/blob/main/Presentation_Slides/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202023-07-23%2011.06.03.png)

