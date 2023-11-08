############################################
#Imputation
###########################################
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)

missingvals <- vroom("./trainWithMissingValues.csv")
GGGtrain <- vroom("./train.csv")
GGGtrain <- GGGtrain %>%
  mutate(type = as.factor(type))
GGGtest <- vroom("./test.csv")

impute_recipe <- recipe(type ~ ., data = missingvals) %>%
  step_impute_linear(has_soul, impute_with = imp_vars(bone_length, rotting_flesh, hair_length)) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul)) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(bone_length, hair_length, has_soul)) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(bone_length, rotting_flesh, has_soul))


prep <- prep(impute_recipe)
baked <- bake(prep, new_data = missingvals)

#check RMSE of imputations
rmse_vec(GGGtrain[is.na(missingvals)], baked[is.na(missingvals)])

##########################################################################
# Build recipe
my_recipe <- recipe(type ~ ., data = GGGtrain)%>%
  step_mutate_at(color, fn = factor) %>%
  step_rm(id) %>%
  step_lencode_glm(color, outcome = vars(type))
  #step_normalize(color)
  
prep <- prep(my_recipe)
baked <- bake(prep, new_data = GGGtrain)

  
  my_mod_RF <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

wf_RF <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_RF)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,(ncol(GGGtrain)-1))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(GGGtrain, v = 5, repeats=1)

## Run the CV
CV_results <- wf_RF %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL


#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('accuracy')

final_wf <- wf_RF %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGtrain)

GGG_prediction_RF <- final_wf %>% predict(new_data = GGGtest, type="class")

submission <- GGG_prediction_RF %>%
  mutate(id = GGGtest$id) %>%
  mutate(type = .pred_class) %>%
  select(2, 3)

vroom_write(submission, "GGGrf.csv", delim = ",")

################################################################
## Naive Bayes
my_recipe_nb <- recipe(type ~ ., data = GGGtrain)%>%
  step_lencode_glm(color, outcome = vars(type))

library(tidymodels)
library(discrim)
library(naivebayes)
## nb model3
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes engine
nb_wf <- workflow() %>%
  add_recipe(my_recipe_nb) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(smoothness(),
                            Laplace(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
## Split data for CV15
folds <- vfold_cv(GGGtrain, v = 5, repeats=1)

## Run the CV
CV_results <- wf_RF %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL


#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('accuracy')

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGtrain)



## Predict

GGG_prediction_NB <- final_wf %>% predict(new_data = GGGtest, type="class")

submission <- GGG_prediction_NB %>%
  mutate(id = GGGtest$id) %>%
  mutate(type = .pred_class) %>%
  select(2, 3)

vroom_write(submission, "GGGnb.csv", delim = ",")


##########################################################
# Neural Networks


nn_recipe <- recipe(formula= type ~ ., data=GGGtrain) %>%
  #update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_lencode_glm(color, outcome = vars(type)) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 250, #or 100 or 250
                activation="relu") %>%
set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
  set_mode("classification")

#for nnet
nn_model <- mlp(hidden_units = tune(),
                epochs = 250) %>%
  set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 100)),
                            levels=5)

folds <- vfold_cv(GGGtrain, v = 5, repeats=1)

## Run the CV
CV_results <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

CV_results %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want


#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('accuracy')

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGtrain)

GGG_prediction_nn <- final_wf %>% predict(new_data = GGGtest, type="class")

submission <- GGG_prediction_nn %>%
  mutate(id = GGGtest$id) %>%
  mutate(type = .pred_class) %>%
  select(2, 3)

vroom_write(submission, "GGGnn.csv", delim = ",")






