################################################################################
# Simulated annealing for random selection of citizens' assemblies
# Translated from R code Version 8.6, Release date: 11.01.2025
# Using simanneal (30.10.2025)
################################################################################

import pandas as pd
import numpy as np
from simanneal import Annealer
from openpyxl import Workbook
import os

#####################################################
# I've cut the R code about loading packages
# and message managment, these are things unique to R
#####################################################

######################
# Utility functions  #
######################

# R code:
# trimws_str <- function(x) {
#   if(is.character(x)) trimws(x) else x
# }

# Python code (trims whitespace): 
def trimws_str(x):
    if isinstance(x, str):
        return x.strip()
    return x

# R code:
# Simple console output function with immediate flushing
# console_and_file_print <- function(message) {
#   cat(message, "\n")
#   flush.console()
# }

# Python code (basically to print things in real time, i think?):
def console_print(message):
    print(message, flush=True)

######################
# Settings variables #
######################

# R code:
# script.dir <- getSrcDirectory(function(x) {x})
# main_directory <- '..'
# input_file_directory <- "data_and_settings"
# file_settings <- TRUE
# settings_filename <- 'settings.xlsx'
# # Set working directory and load settings
# setwd(input_file_directory)
# settings <- read.xlsx(settings_filename, sheet = 1, rowNames = TRUE)

# Python code:
settings_filename = 'settings.xlsx'
# This is what worked, might need to revisit in cleaning up code?
# input_file_directory = "data_and_settings"
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_directory = os.path.join(script_dir, "data_and_settings")

# R code:
# # Extract settings
# settings_list <- list(
#   input_filename = as.character(settings["input_filename", "setting_value"]),
#   par_filename = as.character(settings["par_filename", "setting_value"]),
#   draw_name = as.character(settings["draw_name", "setting_value"]),
#   assembly_size = as.numeric(settings["assembly_size", "setting_value"]),
#   draws_number = as.numeric(settings["draws_number", "setting_value"]),
### Don't have a use for this, not a simanneal input like it was for gensa (I'm pretty sure)
#   SA_script_max_time = as.numeric(settings["SA_max_time", "setting_value"]),
#   SA_temperature = as.numeric(settings["SA_temperature", "setting_value"]),
#   SA_seed = as.numeric(settings["SA_seed", "setting_value"]),
#   SA_max_iterations = as.numeric(settings["SA_max_iterations", "setting_value"]),
### simanneal doesn't have this as an input (I'm pretty sure)
#   SA_nb_stop_imp = as.numeric(settings["SA_nb_stop_imp", "setting_value"]),
#   SA_threshold_stop = as.numeric(settings["SA_threshold_stop", "setting_value"]),
### also not used in simanneal ?
#   SA_max_call = as.numeric(settings["SA_max_call", "setting_value"]),
### is this... play sound when done?
#   finish_signal = as.logical(as.numeric(settings["finish_signal", "setting_value"])),
### whether to write each draw to a seperate file, not dealing with that rn
#   draws_files = as.logical(as.numeric(settings["draws_files", "setting_value"])),
#   household_switch = as.logical(as.numeric(settings["household_duplicate", "setting_value"]))
# )

# Python code (loading settings):
settings = pd.read_excel(os.path.join(input_file_directory, settings_filename), sheet_name=0, index_col=0)
settings_list = {
    'input_filename': str(settings.loc['input_filename', 'setting_value']),
    'par_filename': str(settings.loc['par_filename', 'setting_value']),
    'draw_name': str(settings.loc['draw_name', 'setting_value']),
    'assembly_size': int(settings.loc['assembly_size', 'setting_value']),
    'draws_number': int(settings.loc['draws_number', 'setting_value']),
    'SA_temperature': float(settings.loc['SA_temperature', 'setting_value']),
    'SA_seed': int(settings.loc['SA_seed', 'setting_value']),
    'SA_max_iterations' : int(settings.loc['SA_max_iterations', 'setting_value']),
    'SA_threshold_stop' : int(settings.loc['SA_threshold_stop', 'setting_value']),
    'household_switch': bool(int(settings.loc['household_duplicate', 'setting_value']))
}

#######################################
# Set random seed for reproducibility #
#######################################
#to make sure it's in the range numpy likes
SA_seed = settings_list['SA_seed'] % (2**32) 
np.random.seed(SA_seed)

###########################
# Load and preprocess data#
###########################

# R code:
# # Load and preprocess data with error handling
# load_and_preprocess_data <- function(input_filename, par_filename) {
#   tryCatch({
#     # Load volunteers data
#     volunteers <- read.xlsx(input_filename, sheet = 1, rowNames = TRUE)
#     volunteers_names <- lapply(names(volunteers), trimws_str)
#     volunteers <- data.frame(lapply(volunteers, trimws_str))
#     names(volunteers) <- volunteers_names
    
#     # Load characteristics data
#     characteristics <- read.xlsx(par_filename, sheet = 1)
#     characteristics_names <- lapply(names(characteristics), trimws_str)
#     characteristics <- data.frame(lapply(characteristics, trimws_str))
#     names(characteristics) <- characteristics_names
    
#     # Process characteristics
#     characteristics$counter <- rep(0, nrow(characteristics))
#     characteristics$priority <- sapply(characteristics$priority, function(x) if(as.numeric(x) > 1) as.numeric(x) else 1)
#     characteristics$category <- sapply(characteristics$category, function(x) gsub(" ", ".", x))
    
#     list(
#       volunteers = volunteers,
#       characteristics = characteristics,
#       INPUT_SIZE = nrow(volunteers),
#       categories_number = nrow(characteristics)
#     )
#   }, error = function(e) {
#     stop("Error loading data: ", e$message)
#   })
# }

# Python code:
def load_and_preprocess_data(input_filename, par_filename):
    try:
        # Volunteers
        volunteers = pd.read_excel(os.path.join(input_file_directory, input_filename), sheet_name=0, index_col=0)
        volunteers = volunteers.applymap(trimws_str)

        # Characteristics
        characteristics = pd.read_excel(os.path.join(input_file_directory, par_filename), sheet_name=0)
        characteristics = characteristics.applymap(trimws_str)
        characteristics['counter'] = 0
        characteristics['priority'] = characteristics['priority'].apply(lambda x: max(float(x), 1))
        characteristics['category'] = characteristics['category'].apply(lambda x: x.replace(' ', '.'))

        return volunteers, characteristics, len(volunteers), len(characteristics)
    except Exception as e:
        raise RuntimeError("Error loading data: " + str(e))


# R code:
# # Load data
# data <- load_and_preprocess_data(settings_list$input_filename, settings_list$par_filename)
# volunteers <- data$volunteers
# characteristics <- data$characteristics
# INPUT_SIZE <- data$INPUT_SIZE
# categories_number <- data$categories_number

# Python code:
volunteers, characteristics, INPUT_SIZE, categories_number = load_and_preprocess_data(
    settings_list['input_filename'], settings_list['par_filename']
)

# R code:
# # Pre-compute characteristic vectors for faster access
# char_vectors <- list(
#   category = c(characteristics$category),
#   priority = as.numeric(c(characteristics$priority)),
#   value = as.integer(c(characteristics$value)),
#   feature = c(characteristics$feature)
# )

# Python code (precompute characteristic vectors and feature matrix):
char_vectors = {
    'category': characteristics['category'].tolist(),
    'priority': characteristics['priority'].astype(float).tolist(),
    'value': characteristics['value'].astype(int).tolist(),
    'feature': characteristics['feature'].tolist()
}

# R code:
# # Create feature matrix for faster lookups
# feature_matrix <- matrix(0, nrow = INPUT_SIZE, ncol = categories_number)
# for(k in seq_len(categories_number)) {
#   feature_matrix[, k] <- volunteers[[char_vectors$category[k]]] == char_vectors$feature[k]
# }

# Python code:
feature_matrix = np.zeros((INPUT_SIZE, categories_number), dtype=int)
for k in range(categories_number):
    feature_matrix[:, k] = (volunteers[char_vectors['category'][k]] == char_vectors['feature'][k]).astype(int)

###########################
# Define Simulated Annealer
###########################

# R code:
# # Optimized evaluation function
# evaluation_function <- cmpfun(function(v, b, draw_no, draws_files) {
#   # Pre-compute v indices once
#   v_indices <- (round(v) %% INPUT_SIZE) + 1
  
#   # Fast duplicate check
#   if(length(unique(v_indices)) != length(v_indices)) {
#     return(99999999)
#   }
  
#   # Vectorized household check
#   if(settings_list$household_switch) {
#     household_ids <- volunteers$HOUSEHOLD_ID[v_indices]
#     if(length(unique(household_ids)) != length(v_indices)) {
#       return(99999999)
#     }
#   }
#   # Vectorized feature counting using pre-computed matrix
#   counter <- colSums(feature_matrix[v_indices, , drop = FALSE])
  
#   # Vectorized squared difference calculation
#   ret <- sum(char_vectors$priority * (counter - char_vectors$value)^2)
  
#   if(ret == 0 || b) {
#     characteristics$counter <- characteristics$counter + counter
#     characteristics_summary$counter <- characteristics_summary$counter + counter
    
#     if(draws_files) {
#       drawed_df <- data.frame(
#         No = rownames(volunteers)[v_indices],
#         ID = volunteers$ID[v_indices]
#       )
#       write.xlsx(drawed_df, 
#                  paste0('result_', draw_no, '.xlsx'),
#                  sheetName = paste0('result_', draw_no))
#     }
#   }
  
#   ret
# })

# Python code:
class SortitionAnnealer(Annealer):
    def __init__(self, state, volunteers, feature_matrix, char_vectors, household_switch, threshold_stop, draws_files=False):
        super().__init__(state)
        self.volunteers = volunteers
        self.feature_matrix = feature_matrix
        self.char_vectors = char_vectors
        self.household_switch = household_switch
        self.draws_files = draws_files
        self.INPUT_SIZE = len(volunteers)
        self.threshold_stop = threshold_stop
    
    def move(self):
        idx = np.random.randint(len(self.state))
        self.state[idx] = np.random.randint(0, self.INPUT_SIZE)
    
    def energy(self):
        # Precomputes v indices
        v_indices = np.array(self.state) % self.INPUT_SIZE
        # Duplicate check
        if len(np.unique(v_indices)) != len(v_indices):
            return 99999999
        # Household check
        if self.household_switch:
            household_ids = self.volunteers['HOUSEHOLD_ID'].iloc[v_indices].values
            if len(np.unique(household_ids)) != len(v_indices):
                return 99999999
        # Counts features
        counter = self.feature_matrix[v_indices, :].sum(axis=0)
        # squared difference calculation
        ret = np.sum(np.array(self.char_vectors['priority']) * (counter - np.array(self.char_vectors['value']))**2)
        
        # If energy is 0, write this perfect panel to Excel
        ### I've set draw files to False so this will never happen, i should probably just delete
        if ret == 0 and self.draws_files:
            drawed_df = pd.DataFrame({
                'No': self.volunteers.index[v_indices],
                'ID': self.volunteers['ID'].iloc[v_indices]
            })
            drawed_df.to_excel(f'result_{self.draw_no}.xlsx', index=False)

        # Early stop if energy below threshold
        ### soooo... in gensa there is a threshold stop and in simanneal there isn't
        ### rn I'm going to manually force it to stop.
        self.last_energy = ret  # save it?
        if ret <= self.threshold_stop:
            raise ThresholdReached(f"Threshold reached: {ret}")

        return ret
    
# Custom exception to stop annealer early
class ThresholdReached(Exception):
    pass

#################################################################
# There is some analysis setups here that I'm ignoring
# I'm also not replicating any of the inner loop analyis stuff rn
#################################################################

#############
# Run draws #
#############

# R code:
# # Main drawing loop
# for(draw_no in 1:settings_list$draws_number) {
#   console_and_file_print(paste("Starting draw no:", draw_no, 'of', settings_list$draws_number))
  
#   tryCatch({
#     characteristics$counter <- rep(0, categories_number)
    
#     # Initialize with random selection of N unique volunteers
#     initial_params <- sample(1:INPUT_SIZE, settings_list$assembly_size, replace = FALSE)
    
#     RESULT <- GenSA(
#       par = as.numeric(initial_params),
#       fn = evaluation_function,
#       lower = as.numeric(rep(1, settings_list$assembly_size)),
#       upper = as.numeric(rep(INPUT_SIZE, settings_list$assembly_size)),
#       b = FALSE,
#       draw_no = draw_no,
#       draws_files = settings_list$draws_files,
#       control = list(
#         max.time = settings_list$SA_script_max_time,
#         smooth = TRUE,
#         verbose = FALSE,
#         temperature = settings_list$SA_temperature,
#         threshold.stop = settings_list$SA_threshold_stop,
#         max.call = settings_list$SA_max_call,
#         maxit = settings_list$SA_max_iterations,
#         nb.stop.improvement = settings_list$SA_nb_stop_imp,
#         seed = settings_list$SA_seed
#       )
#     )
#     # Update results summary
#         results_summary[paste0('draw_', draw_no)] <- rep(0, nrow(volunteers))
#         selected_ids <- volunteers$ID[selected_indices]
#         for(id in selected_ids) {
#         results_summary[as.character(id), paste0('draw_', draw_no)] <- 
#             results_summary[as.character(id), paste0('draw_', draw_no)] + 1
#         results_summary[as.character(id), 'drawn_counter'] <- 
#             results_summary[as.character(id), 'drawn_counter'] + 1
#         }
# }

assembly_size = settings_list['assembly_size']
draws_number = settings_list['draws_number']
household_switch = settings_list['household_switch']
threshold_stop = settings_list['SA_threshold_stop']
#can i put random seed here?

results_summary = pd.DataFrame({
    'volunteers': volunteers['ID'].astype(str),
    'drawn_counter': 0
})
results_summary.set_index('volunteers', inplace=True)

console_print(f"Drawing {draws_number} panel(s) for: {settings_list['draw_name']}")

# Loop over draw number
for draw_no in range(1, draws_number + 1): 
    # Pring progress messages
    console_print(f"Starting draw {draw_no} of {draws_number}")
    initial_panel = np.random.choice(INPUT_SIZE, assembly_size, replace=False)
    annealer = SortitionAnnealer(initial_panel, volunteers, feature_matrix, char_vectors, household_switch, threshold_stop)
    annealer.steps = settings_list['SA_max_iterations']
    annealer.Tmax = settings_list['SA_temperature']
    annealer.Tmin = 0.01
    # Do the actual simulated annealing
    # best_state, best_energy = annealer.anneal()
    # I'm wrapping it in this thing so that the threshold reached only stops current draw
    try:
        best_state, best_energy = annealer.anneal()
    except ThresholdReached as e:
        console_print(f"Threshold reached in draw {draw_no}: {e}")
        best_state = annealer.state
        best_energy = annealer.last_energy  
    
    selected_indices = np.array(best_state) % INPUT_SIZE
    results_summary[f'draw_{draw_no}'] = 0
    results_summary.iloc[selected_indices, results_summary.columns.get_loc(f'draw_{draw_no}')] = 1
    results_summary['drawn_counter'] += results_summary[f'draw_{draw_no}']

###########################
# Save results
###########################

# R code:
# Set results filename
# xlsx_results_filename <- paste0(
#   "Results-", 
#   settings_list$draw_name, 
#   "-", 
#   settings_list$draws_number, 
#   if(settings_list$draws_number == 1) "-draw.xlsx" else "-draws.xlsx"
# )
# # Results processing and saving
# draws_times <- data.frame(t(draws_times))
# function_value <- data.frame(t(function_value))

# for (i in 1:ncol(draws_times)) {
#   newColName <- paste('time_no', i, sep = "_")
#   colnames(draws_times)[i] <- newColName
#   colnames(function_value)[i] <- newColName
# }

# Python code:
#TODO: save it in folder lol
xlsx_results_filename = f"Results-{settings_list['draw_name']}-{draws_number}-draws.xlsx"

results_summary.to_excel(xlsx_results_filename)
console_print(f"Saved panel to {xlsx_results_filename}")

