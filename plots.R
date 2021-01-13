library(tidyverse)

data <- read_csv('results.csv')

data <- na.omit(data)

data %>%
  ggplot(aes(a, fill=status)) +
  geom_bar(position = "dodge")
