#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import glob
import sys


from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFECV

from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tabulate import tabulate

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score, accuracy_score

import pickle

warnings.filterwarnings("ignore")


# # # Read the data from Empathy dataset

# In[2]:


Files = glob.glob('E:/University of Essex/DataScience/Re-assesment/EyeT/*.csv')

readFile_count = len(Files)
print ("Eye-Tracking Data:", readFile_count);


# In[4]:


def preprocess_data(data):
     # Drop the first column
    data = data.iloc[:, 1:]
    
    # List of columns to drop which not important for Empathy
    drop_column = ['Mouse position X', 'Mouse position Y', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
                    'Event', 'Event value',
                    'Computer timestamp', 'Export date', 'Recording date',
                    'Recording date UTC', 'Recording start time', 'Timeline name', 'Recording Fixation filter name',
                    'Recording software version', 'Recording resolution height', 'Recording resolution width',
                    'Recording monitor latency', 'Presented Media width', 'Presented Media height',
                    'Presented Media position X (DACSpx)', 'Presented Media position Y (DACSpx)', 'Original Media width',
                    'Recording start time UTC', 'Original Media height', 'Sensor']
    
   
    data[['Pupil diameter left', 'Pupil diameter right', 'Fixation point X', 'Fixation point Y']] = \
        data[['Pupil diameter left', 'Pupil diameter right', 'Fixation point X', 'Fixation point Y']].ffill()
    
    
    numeric_columns = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z',
                    'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z',
                    'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
                    'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
                    'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
                    'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
                    'Gaze point left X (MCSnorm)', 'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
                    'Gaze point right Y (MCSnorm)', 'Pupil diameter left', 'Pupil diameter right']
    

    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col].str.replace(',', '.'), errors='coerce')

    return data


# In[6]:


def summarizing_eye_tracking_data(data, group):
    # Filter valid eye tracking data
    valid_data = data[(data['Validity left'] == 'Valid') & (data['Validity right'] == 'Valid')]

    # Count total fixations
    total_fixations = data[data['Eye movement type'] == 'Fixation'].shape[0]

    # Calculate average fixation duration
    avg_duration = data[data['Eye movement type'] == 'Fixation']['Gaze event duration'].mean()

    # Calculate statistics for different columns
    pupil_diameter = data[['Pupil diameter left', 'Pupil diameter right']].mean(axis=1).agg(['mean', 'median', 'std']).rename(lambda x: f'Pupil Diameter {x.capitalize()}')
    gaze_point_x = data['Gaze point X'].agg(['mean', 'median', 'std']).rename(lambda x: f'Gaze Point X {x.capitalize()}')
    gaze_point_y = data['Gaze point Y'].agg(['mean', 'median', 'std']).rename(lambda x: f'Gaze Point Y {x.capitalize()}')
    fixation_point_x = data['Fixation point X'].agg(['mean', 'median', 'std']).rename(lambda x: f'Fixation Point X {x.capitalize()}')
    fixation_point_y = data['Fixation point Y'].agg(['mean', 'median', 'std']).rename(lambda x: f'Fixation Point Y {x.capitalize()}')

    summary_data = {
        'Participant Name': data['Participant name'].iloc[0],
        'Project Name': group,
        'Recording Name': data['Recording name'].iloc[0],
        'Total Fixations': total_fixations,
        'Avg. Fixation Duration': avg_duration
    }
    summary_data.update(pupil_diameter)
    summary_data.update(gaze_point_x)
    summary_data.update(gaze_point_y)
    summary_data.update(fixation_point_x)
    summary_data.update(fixation_point_y)

    summary = pd.DataFrame(summary_data, index=[0])
    

    return summary


# # # Pipeline

# In[7]:


# Initialize an empty list to store processed Datasets
processed_data_list = []
current_iteration = 0


# Loop through each filename and process the data
for filename in Files:
    # Read the raw data from the CSV file
    raw_data = pd.read_csv(filename, usecols=lambda column: column != 0, low_memory=True)
     
    
    processed_data = preprocess_data(raw_data)
    
   
    file_name = os.path.basename(filename)
       
    # Determine the group based on the file name
    if file_name.startswith('EyeT_group_dataset_III_'):
        group = 'Test group experiment'
    elif file_name.startswith('EyeT_group_dataset_II_'):
        group = 'Control group experiment'
    
    
    summary = summarizing_eye_tracking_data(processed_data, group)
    processed_data_list.append(summary)
    
    current_iteration += 1
    
# Concatenate all the summaries into a single Datasets
print("Data summarization successful")
final_processed_df = pd.concat(processed_data_list, ignore_index=True)


# In[8]:


final_processed_df.head()


# # ## Exploratory Data Analysis (EDA)

# In[9]:


pd.set_option('display.max_rows', None)

final_processed_df.head()


# In[10]:


final_processed_df.isna().sum()


# ### Get Count of CONTROL GROUP and TEST GROUP experiments

# In[11]:


counts = final_processed_df['Project Name'].value_counts()

control_group_count = counts.get("Control group experiment", 0)
test_group_count = counts.get("Test group experiment", 0)

print(f' "Total Control Group Experiment": {control_group_count}')
print(f' "Total Test Group Experiment": {test_group_count}')


# In[12]:


final_processed_df.describe()


# In[13]:


final_processed_df.info()


# In[14]:


final_processed_df.shape


# ## #Load Questionare dataset to read Empathy score

# Here the goal is the empathy score.The total score extended will be used because it includes more questions that are more accurate in measuring empathy. 

# In[15]:


df_questionare = pd.read_csv('E:/University of Essex/DataScience/Re-assesment/Empathy_Score/Questionnaire_datasetIA.csv', encoding='ISO-8859-1')


# In[16]:


df_questionare.head()


# In[17]:


df_questionare.describe()


# ## # Merge both dataset into a single with an empathy target score

# In[112]:


merged_result = pd.concat([final_processed_df, df_questionare[['Participant nr', 'Total Score extended']]], axis=1)

merged_result.drop(columns=['Participant nr'], inplace=True)

print("Successfully merged dataframes.")


# In[113]:


merged_result.head()


# In[114]:


merged_result.describe()


# In[115]:


merged_result.shape


# In[116]:


merged_result.info()


# # analyzing how the eye-tracking features relate to empathy.

# In[24]:


# Create a scatter plot with different colors for each project
unique_projects = merged_result['Project Name'].unique()
plot_colors = ['blue', 'red', 'orange'] 

fig, ax = plt.subplots()

# Iterate through each unique project and plot its data
for project, color in zip(unique_projects, plot_colors):
    project_data = merged_result[merged_result['Project Name'] == project]
    
    print(f"Processing project: {project}")
    print(f"Number of data points: {len(project_data)}")
    ax.scatter(project_data['Pupil Diameter Mean'], project_data['Total Score extended'], c=color, label=project, zorder=3)

ax.set_xlabel('Pupil Diameter Mean')
ax.set_ylabel('Empathy Score')
ax.set_title('Relation Between Pupil Diameter and Empathy Score by Project')

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()


# In[25]:


# Create a DataFrame to store the table data
table_data = []

# Create a list of unique projects
unique_projects = merged_result['Project Name'].unique()

# Iterate through each unique project and gather data for the table
for project in unique_projects:
    project_data = merged_result[merged_result['Project Name'] == project]
    
    num_data_points = len(project_data)
    mean_pupil_diameter = project_data['Pupil Diameter Mean'].mean()
    mean_empathy_score = project_data['Total Score extended'].mean()
    
    table_data.append([project, num_data_points, mean_pupil_diameter, mean_empathy_score])

# Create a DataFrame from the collected table data
table_df = pd.DataFrame(table_data, columns=['Project Name', 'Number of Data Points', 'Mean Pupil Diameter', 'Mean Empathy Score'])

# Convert the DataFrame to a tabulated format
table_str = tabulate(table_df, headers='keys', tablefmt='grid')

# Display the table
print(table_str)


# In[26]:


# Create a scatter plot with different colors for each project
unique_projects = merged_result['Project Name'].unique()
colors = ['blue', 'red'] 

fig, ax = plt.subplots()

for project, color in zip(unique_projects, colors):
    project_data = merged_result[merged_result['Project Name'] == project]
    ax.scatter(project_data['Total Fixations'], project_data['Total Score extended'], c=color, label=project)

# Add labels and title
ax.set_xlabel('Total Fixations')
ax.set_ylabel('Empathy Score')
ax.set_title('Total Fixations vs Empathy Score by Project')

ax.legend()

plt.show()


# ## # standard deviations and dimensions of each participant's trails

# In[27]:


# Get unique participant names
unique_participants = merged_result['Participant Name'].unique()

selected_participants = unique_participants[:6]

# Create subplots
fig, axs = plt.subplots(nrows=len(selected_participants), figsize=(10, 6*len(selected_participants)))
plt.subplots_adjust(hspace=0.5) 

#Initialize a table for storing data
table_data = []

for i, participant in enumerate(selected_participants):
    participant_data = merged_result[merged_result['Participant Name'] == participant]
    participant_data = participant_data.reset_index().rename(columns={'index': 'occurrence'}).head(6)

    grouped_data = participant_data.groupby('occurrence').agg({'Pupil Diameter Mean': 'mean', 'Pupil Diameter Median': 'mean', 'Pupil Diameter Std': 'mean'}).reset_index()

    ax = axs[i]

    # Create a bar plot showing mean and median with error bars for standard deviation
    x = np.arange(len(grouped_data))
    width = 0.35

    ax.bar(x - width/2, grouped_data['Pupil Diameter Mean'], width, label='Mean', yerr=grouped_data['Pupil Diameter Std'], capsize=5, color='orange', alpha=0.7)
    ax.bar(x + width/2, grouped_data['Pupil Diameter Median'], width, label='Median', color='green', alpha=0.7)


    ax.set_xlabel('Occurrence')
    ax.set_ylabel('Avg Pupil Diameter (mm)')
    ax.set_title(f'Mean and Median for {participant}')

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(grouped_data['occurrence'])

    # Add legend
    ax.legend()
    
    # Store data for the table
    table_data.append([participant] + grouped_data['Pupil Diameter Mean'].tolist() + grouped_data['Pupil Diameter Median'].tolist())


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Create a table using tabulate
table_headers = ['Participant', 'Occurrence 1 Mean', 'Occurrence 2 Mean', 'Occurrence 3 Mean', 'Occurrence 4 Mean', 'Occurrence 5 Mean', 'Occurrence 6 Mean',
                 'Occurrence 1 Median', 'Occurrence 2 Median', 'Occurrence 3 Median', 'Occurrence 4 Median', 'Occurrence 5 Median', 'Occurrence 6 Median']

table_df = pd.DataFrame(table_data, columns=table_headers)


print(table_df)


# ## # Assess control and test groups separately

# In[28]:


# Create a dictionary to store the DataFrames
project_data_frames = {}

# Iterate over each unique project name
for project_name in unique_projects:
    
    filtered_data = merged_result[merged_result['Project Name'] == project_name]

    
    project_data_frames[project_name] = filtered_data


# In[73]:


target_project_name = 'Control group experiment'
target_project_control_dataframe = project_data_frames[target_project_name]


target_project_dataframe.head()


# In[76]:


# Creating a new DataFrame with relevant columns
selected_columns = ['Participant Name', 'Project Name', 'Pupil Diameter Std', 'Pupil Diameter Mean', 'Pupil Diameter Median', 'Recording Name', 'Total Score extended']
pupil_control_data = target_project_control_dataframe[selected_columns].copy()

# Display the initial rows of the newly created DataFrame
pupil_control_data.head()


# ## # Split control group data using StratifiedKFold

# In[77]:


# Load the dataset and separate features and target
input_data = target_project_control_dataframe
features = input_data.drop(['Project Name', 'Total Score extended'], axis=1)
target = input_data['Total Score extended']
group_labels = input_data['Project Name']  # Used for stratification

# Reset indices of features and target DataFrames
features.reset_index(drop=True, inplace=True)
target.reset_index(drop=True, inplace=True)
group_labels.reset_index(drop=True, inplace=True)


# Define the number of splits for cross-validation
num_splits = 5

# Initialize the StratifiedKFold cross-validator
stratified_kfold = StratifiedKFold(n_splits=num_splits)

# Iterate over the cross-validation splits
for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(features, group_labels)):
    # Get the training and testing data for this fold
    X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[78]:


target_project_name = 'Test group experiment'
target_project_test_dataframe = project_data_frames[target_project_name]


target_project_test_dataframe.head()


# ## # Split test group data using StratifiedKFold

# In[79]:


# Load the dataset and separate features and target
input_data = target_project_test_dataframe
features = input_data.drop(['Participant Name', 'Total Score extended'], axis=1)
target = input_data['Total Score extended']
group_labels = input_data['Participant Name']  # Used for stratification

# Check for missing values in the target array
print("Number of missing values in target:", target.isnull().sum())

# Define the number of splits for cross-validation
num_splits = 5

# Initialize the StratifiedKFold cross-validator
stratified_kfold = StratifiedKFold(n_splits=num_splits)

# Iterate over the cross-validation splits
for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(features, group_labels)):
    # Get the training and testing data for this fold
    X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # # Assess another dataset for pupil

# Empathy analysis may be dependent on pupil features. This approach involves focusing on the pupil to predict empathy and ignoring other features that may be less relevant.

# In[80]:


# Creating a new DataFrame with relevant columns
selected_columns = ['Participant Name', 'Project Name', 'Pupil Diameter Std', 'Pupil Diameter Mean', 'Pupil Diameter Median', 'Recording Name', 'Total Score extended']
pupil_test_data = target_project_test_dataframe[selected_columns].copy()

# Display the initial rows of the newly created DataFrame
pupil_test_data.head()


# ## # correlation matrix for control group experiment and test group experiment

# In[89]:


def plot_correlation_heatmap(data_frame, target_feature, num_features=15):
    
    corr_matrix = data_frame.corr()

    # Select the top_n columns with the highest correlation
    cols = corr_matrix.nlargest(num_features, target_feature)[target_feature].index

    cm = data_frame[cols].corr()

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(cm, annot=True, cmap='viridis')
    ax.set_title('Top Correlation Heatmap for ' + target_feature)

    plt.show()
    
    return


# In[91]:


plot_correlation_heatmap(target_project_dataframe, 'Total Score extended', num_features=20)


# In[92]:


plot_correlation_heatmap(target_project_dataframe, 'Total Score extended', num_features=20)


# ## # Classification Model

# Train and evaluate the control group to predict empathy score.

# In[36]:


def train_and_evaluate_model(data_frame, group_label, model_type='regressor'):
    # Remove rows with missing target values
    data_frame = data_frame.dropna(subset=['Total Score extended'])
    
    # Extract features (X) and target (y) variables
    X_features = data_frame.drop(columns=['Total Score extended', 'Project Name', 'Recording Name'])
    y_target = data_frame['Total Score extended']
    
    # Initialize an empty DataFrame to store results
    evaluation_results = pd.DataFrame(columns=['Participant Name', 'Original Empathy Score', 'Predicted Empathy Score'])
    
    # Encode participant names using LabelEncoder
    participant_encoder = LabelEncoder()
    X_features['Participant Name'] = participant_encoder.fit_transform(X_features['Participant Name'])
    participant_groups = data_frame['Participant Name']
    
    # Set the number of splits equal to the number of unique participant groups
    n_splits = 8
    group_kfold = GroupKFold(n_splits=n_splits)
    
    
    # Lists to store evaluation metrics
    mse_scores = []
    r2_scores = []
    rmse_scores = []
    medae_scores = []
    y_test_all = []
    y_pred_all = []
    
     # Determine the model type and initialize the model
    if model_type == 'regressor':
        model = RandomForestRegressor()
    elif model_type == 'classifier':
        model = DecisionTreeClassifier()
    elif model_type == 'logistic':
        model = LogisticRegression()
    else:
        raise ValueError("Invalid model type. Choose 'regressor' or 'classifier'.")
    
    
    # Iterate through folds using GroupKFold
    for fold, (train_index, test_index) in enumerate(group_kfold.split(X_features, y_target, groups=participant_groups)):
        X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]
        
        # Initialize and train the model (RandomForestRegressor)
#         model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        
        print(f"Fold {fold + 1}:")
        
        for idx, (participant, original_score) in enumerate(zip(data_frame.iloc[test_index]['Participant Name'], y_test)):
            predicted_score = y_pred[idx]
            print(f"  Participant Name: {participant}, Original Empathy Score: {original_score}, Predicted Empathy Score: {predicted_score:.2f}")
            
            # Store the results for this fold
            evaluation_results = evaluation_results.append({'Participant Name': participant,
                                                            'Original Empathy Score': original_score,
                                                            'Predicted Empathy Score': predicted_score,},
                                                           
                                                           ignore_index=True)
        
        # Calculate evaluation metrics for this fold
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        
        # Store metrics in lists for averaging later
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        medae_scores.append(medae)
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # Calculate average evaluation metrics
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_medae = np.mean(medae_scores)
        
    
    print(f"Average Root Mean Squared Error: {avg_rmse}")
    print(f"Average R-squared: {avg_r2}")
    print(f"Average Median Absolute Error: {avg_medae}")
    
    
    return evaluation_results


# In[38]:


print("*** Train and Evaluate Model using RandomForestRegressor on Control Group***")
target_project_control_evaluation = train_and_evaluate_model(target_project_dataframe, "Control Group", model_type='regressor')
target_project_control_evaluation.head()


# In[39]:


print("*** Train and Evaluate Model using DecisionTreeClassifier on Control Group***")
results_control_classifier = train_and_evaluate_model(target_project_dataframe, "Control Group", model_type='classifier')
target_project_control_evaluation.head()


# In[40]:


print("*** Train and Evaluate Model using LogisticRegression on Control Group***")
target_project_control_evaluation = train_and_evaluate_model(target_project_dataframe, "Control Group", model_type='logistic')
target_project_control_evaluation.head()


# In[43]:


target_project_control_evaluation.info()


# In[44]:


def actual_vs_predicted_plot(predictions_dataframe,title):
    true_scores = predictions_dataframe['Original Empathy Score'].tolist()
    predicted_scores = predictions_dataframe['Predicted Empathy Score'].tolist()

    plt.scatter(true_scores, predicted_scores, color='green', label='Predicted')
    plt.xlabel('Original Empathy Score')
    plt.ylabel('Predicted Empathy Scores')
    plt.title(f'Actual vs. Predicted Empathy Scores ({title})')

    # Add a line representing perfect prediction
    min_value = min(min(true_scores), min(predicted_scores))
    max_value = max(max(true_scores), max(predicted_scores))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', label='Perfect Prediction')

    plt.legend()
    plt.show()
    
    return


# In[45]:


actual_vs_predicted_plot(target_project_control_evaluation, 'Control Group')


# Train and evaluate the test group to predict empathy score.

# In[46]:


target_project_test_evaluation = train_and_evaluate_model(target_project_dataframe, "Test Group", model_type='regressor')
target_project_test_evaluation.head()


# In[47]:


target_project_test_evaluation = train_and_evaluate_model(target_project_dataframe, "Test Group", model_type='classifier')
target_project_test_evaluation.head()


# In[48]:


target_project_test_evaluation = train_and_evaluate_model(target_project_dataframe, "Test Group", model_type='logistic')
target_project_test_evaluation.head()


# In[49]:


actual_vs_predicted_plot(target_project_test_evaluation, 'Test Group')


# ## # Empathy Prediction Outcomes

# Calculate the average empathy score for each participant's determine overall predicted empathy.

# In[60]:


def visualize_empathy_scores_summary(data_df):
    # Calculate the mean of original and predicted empathy scores for each participant
    summary_df = data_df.groupby('Participant Name').agg({
        'Original Empathy Score': 'first',
        'Predicted Empathy Score': 'mean'
    })

    # Display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("Participant-wise Summary DataFrame:")
    print(summary_df.to_markdown(tablefmt="grid"))

    # Reshape the summary dataframe for visualization
    melted_df = summary_df.reset_index().melt(
        id_vars=['Participant Name'],
        value_vars=['Original Empathy Score', 'Predicted Empathy Score'],
        var_name='Score_Type',
        value_name='Score'
    )

    # Display scores for a subset of participants
    first_n_participants = melted_df['Participant Name'].unique()[:7]
    filtered_df = melted_df[melted_df['Participant Name'].isin(first_n_participants)]

    
    # Set a custom color palette
    colors = ['#b41f4e', '#601fb4']
    
    # Create a bar plot to visualize original and predicted scores
    plt.figure(figsize=(10, 5))
    sns.pointplot(data=filtered_df, x='Participant Name', y='Score', hue='Score_Type', palette=colors, markers=['o', 's'])

    plt.title('Comparison of Actual and Predicted Empathy Scores for Select Participants')
    plt.xlabel('Participant Name')
    plt.ylabel('Empathy Score')

    plt.show()

    return


# In[61]:


visualize_empathy_scores_summary(target_project_test_evaluation)


# In[62]:


visualize_empathy_scores_summary(target_project_control_evaluation)

