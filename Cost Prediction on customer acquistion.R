# Cost Prediction on Customer Acquisition

# Importing libraries

library(tidyverse)
library(readr)
library(reshape)
library(reshape2)
library(tidymodels)
library(vip)
library(recipes)

# Loading the data

customer_data <- read_csv("media prediction and its cost.csv")
attach(customer_data)

# Data cleaning and EDA
## Data cleaning

sum(is.na(customer_data))  

skimr::skim(customer_data)

### Evaluating the categorical variables

customer_data %>%
  count(food_category,sort = T)  

customer_data %>% 
  count(food_department,sort = T)

customer_data %>%
  count(brand_name,sort = T)  

customer_data %>%
  count(promotion_name,sort = T) 

customer_data %>%
  count(store_city,sort = T)  

customer_data %>%
  count(store_state,sort = T) 

customer_data %>%
  count(media_type,sort = T) 

### Distribution of the response variable (cost)

ggplot(customer_data,aes(x= cost)) +
  geom_histogram()   

## Visualizing correlations

### dropping the character variables

customer_numeric <- customer_data[sapply(customer_data,is.numeric)]

### Correlation matrix and heatmap
correlation_mat <- round(cor(customer_numeric),2)

melted_correlation_mat <- melt(correlation_mat)

ggplot(data = melted_correlation_mat, aes(x=Var1, y=Var2,
                                   fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),
            color = "black", size = 4) +
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1))

## Dropping highly correlated numeric variables to reduce multi-collinearity

customer_data <- customer_data %>% 
  select(-c(salad_bar,`avg_cars_at home(approx)...19`,gross_weight,meat_sqft,
            frozen_sqft,`store_sales(in millions)`,`store_cost(in millions)`))

## Dropping non-informative categorical columns
customer_data <- customer_data %>%
  select(-c(recyclable_package,low_fat,food_department,food_family,food_category,
            gender,marital_status,`avg. yearly_income`,education,member_card,
            houseowner,sales_country,occupation,store_city))

# Model Building

## Converting character variables to factors

customer_data <- customer_data %>%
  mutate_if(is_character,factor)


## Splitting the data into training and testing

set.seed(123)

split <- initial_split(customer_data,prop = 0.7)
customer_train <- training(split)
customer_test <- testing(split)

## Creating a recipe and collapsing categorical variables with many categories 

customer_recipe <- recipe(cost~.,data = customer_train) %>%
  step_other(store_state,threshold = 0.08) %>%
  step_other(promotion_name,threshold = 0.035) %>%
  step_other(brand_name,threshold = 0.026)

customer_prep <- prep(customer_recipe)
juiced <- juice(customer_prep)

## Random Forest specification model

tune_spec <- rand_forest(
  mtry = tune(),
  trees = 100,
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("ranger")

## Adding a workflow

tune_workflow <- workflow() %>%
  add_recipe(customer_recipe) %>%
  add_model(tune_spec)


## Training the hyperparameters

set.seed(389)
customer_folds <- vfold_cv(customer_train)

doParallel::registerDoParallel()

set.seed(687)
tune_result <- tune_grid(tune_workflow,
                           resamples = customer_folds,
                           grid= 10)

tune_result %>%
  collect_metrics()

## Tabulating the metrics 
tune_result %>%
  collect_metrics() %>%
  filter(.metric=="rmse") %>%
  select(mean,min_n,mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter")

## Graphical visualization of the tuned parameters
tune_result %>%
  collect_metrics() %>%
  filter(.metric=="rmse") %>%
  select(mean,min_n,mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value,mean,color = parameter)) +
  geom_point(show.legend = F) +
  facet_wrap(~parameter,scales = "free_x") + 
  labs(x= NULL,y="rmse")

## Using the range of values obtained from the graph in the regular grid

rf_grid <- grid_regular(
  mtry(range = c(7,15)),
  min_n(range = c(14,30)),
  levels = 5
)


## Performing the tuning process using the grid obtained in the above process.

set.seed(898)

regular_result <- tune_grid(tune_workflow,
                         resamples = customer_folds,
                         grid= rf_grid)

## Choosing the best model and finalizing the results.

best_rmse <- select_best(regular_result,"rmse")

final_rf_model <- finalize_model(tune_spec,best_rmse)

## Analyzing variable importance

final_rf_model %>%
  set_engine("ranger",importance = "permutation") %>%
  fit(cost~.,
      data = juice(customer_prep)) %>%
  vip(geom = "point")

## Final workflow

final_wf <- workflow() %>%
  add_recipe(customer_recipe) %>%
  add_model(final_rf_model)

final_result <- final_wf %>%
  last_fit(split)

final_result %>%
  collect_metrics()



























