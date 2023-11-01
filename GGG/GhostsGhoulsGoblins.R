############################################
#Imputation
###########################################
library(tidymodels)
library(tidyverse)
library(vroom)

missingvals <- vroom("./trainWithMissingValues.csv")
GGGtrain <- vroom("./train.csv")

impute_recipe <- recipe(type ~ ., data = missingvals) %>%
  step_impute_linear(has_soul, impute_with = imp_vars(bone_length, rotting_flesh, hair_length)) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul)) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(bone_length, hair_length, has_soul)) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(bone_length, rotting_flesh, has_soul))


prep <- prep(impute_recipe)
baked <- bake(prep, new_data = missingvals)

rmse_vec(GGGtrain[is.na(missingvals)], baked[is.na(missingvals)])
