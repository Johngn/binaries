library(tidyverse)
data(iris)

data <- read_csv('results.csv')

data %>%
  ggplot(aes(a, fill=status)) +
  geom_bar(position = "dodge")
