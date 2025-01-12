import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas openpyxl
import numpy as np
import plotly.graph_objects as go
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Oil Analysis", page_icon=":bar_chart:", layout="wide")

# ---- MAINPAGE ----
st.title(":bar_chart: 🏭 Oil Analysis")
st.markdown("##")
st.markdown("""
This app performs simple oil analysis with prediction!
* **Python libraries:** plotly, pandas, streamlit, matplotlib.
""")

# ---- READ EXCEL ----
# @st.cache(allow_output_mutation=True)
def get_data_from_csv():
    df = pd.read_csv(
        "OilSample_test.csv",  # Path to the CSV file
        sep=";",
        nrows=1000  # Limit to 1000 rows
    )
    return df

df = get_data_from_csv()

st.write('Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
# Sample data structure with additional variables

data_description = {
    "serialno": "Equipment serial number",
    "sampledate": "Date the sample was taken",
    "evalcode": "Oil status code given after human evaluation. (target variable)",
    "compartid": "ID of compartment oil sample is taken from.",
    "oiltypeid": "Make and type of oil in sample",
    "oilgradeid": "Grade of oil in sample",
    "oilhours": "Age of oil in sample in machine hours",
    "machinehours": "Machine age in hours at time of sampling",
    "PQI": "Particle Quantification Index",
    "Fe": "Iron",
    "Cu": "Copper",
    "Cr": "Chrome",
    "Pb": "Lead",
    "Sn": "Tin",
    "Ni": "Nickel",
    "Al": "Aluminium",
    "Si": "Silicon",
    "Na": "Sodium",
    "K": "Potassium",
    "Mo": "Molybdenum",
    "B": "Boron",
    "Ba": "Barium",
    "Mg": "Magnesium",
    "Ca": "Calcium",
    "Zn": "Zinc",
    "P": "Phosphorus",
    "Ag": "Silver",
    "Mn": "Manganese",
    "V": "Vanadium",
    "Ti": "Titanium",
    "Cd": "Cadmium",
    "BO3": "Borate",
    "PO4": "Phosphate",
    "H2O": "Water content",
    "F": "Fluor",
    "V40": "Viscosity at 40 deg C",
    "OXI": "Oxide content",
    "NIT": "Nitrate content",
    "SUL": "Sulphate content",
    "ISO6": "ISO 6 Oil Cleanliness",
    "ISO14": "ISO 14 Oil Cleanliness",
    "X6": "Particle count > 6 micron",
    "X10": "Particle count > 10 micron",
    "X14": "Particle count > 14 micron",
    "X21": "Particle count > 21 micron",
    "X25": "Particle count > 25 micron",
    "X38": "Particle count > 38 micron",
    "X70": "Particle count > 70 micron"
}

# Create a DataFrame
df_desc = pd.DataFrame(data_description.items(), columns=["Variable", "Description"])

# Streamlit app
st.subheader("Variables Details")

# Create an expander for variable descriptions
with st.expander("View Variable Descriptions"):
    st.table(df_desc)

st.subheader("Oil Types Details")
evalcode_counts= df['evalcode'].value_counts().size
st.markdown("""
There are **4 types of oil evalcode** that can be assigned to the sample:
* **A:** Oil properties are within acceptable limits and operation can continue as usual,
* **B:** Certain results are outside acceptable ranges, minor problems with machinery,
* **C:** Unsatisfactory results are present, significant problem with the compartment and lubricant properties,
* **X:** Clear contamination needing immediate diagnostic and corrective action to prevent possible failure.
""")
st.write('Basic information about ' + str(evalcode_counts)+ ' types of oil evalcode  in the dataset.')

# Oil evalcode types in dataset
evalcode_stat = df['evalcode'].value_counts()

fig_evalcode_stat = px.bar(y=evalcode_stat.values,
             x=evalcode_stat.index,
             color = evalcode_stat.index,
             color_discrete_sequence=px.colors.sequential.OrRd,
             text=evalcode_stat.values,
             title="<b>Oil Types in the Sample</b>")

fig_evalcode_stat.update_layout(
    xaxis_title="Evalcode Types",
    yaxis_title="Count",
    font = dict(size=14,family="Franklin Gothic"))

fig_evalcode_stat.update_layout(legend=dict(
    title="Evalcode Types"
))
st.plotly_chart(fig_evalcode_stat)

st.markdown("""We can notice that **A** and **B** oil type are top frequent types in the sample.""")
st.subheader("Dataset - General Info")

# Capture the output of df.info()
df_info = pd.DataFrame({
    "Column Name": df.columns,  # Nazwy kolumn
    "Non-Null Count": [df[col].notnull().sum() for col in df.columns],  # Liczba wartości nie-null dla każdej kolumny
    "Dtype": [str(df[col].dtype) for col in df.columns]  # Typ danych każdej kolumny
})
#Dataset Dataframe
st.dataframe(df)
# Display the DataFrame info in Streamlit
st.write("Dataset Info:")

dtype_stat = df_info['Dtype'].value_counts()

fig_dtype_stat = px.bar(y=dtype_stat.values,
             x=dtype_stat.index,
             color = dtype_stat.index,
             color_discrete_sequence=px.colors.sequential.OrRd,
             text=dtype_stat.values,
             title="<b>Data Types in the Sample</b>")

fig_dtype_stat.update_layout(
    xaxis_title="Data Types",
    yaxis_title="Count",
    font = dict(size=14,family="Franklin Gothic"))

fig_dtype_stat.update_layout(legend=dict(
    title="Data Types"
))

left_column,  middle_column, right_column = st.columns(3)
left_column.subheader("Data Types Chart")
left_column.plotly_chart(fig_dtype_stat)
right_column.subheader("Data Info Table")
right_column.dataframe(df_info)



st.markdown("""
Main Key points:
* **Data types:** The dataset contains three main data types: object, float, and integer.
* **Object type:** Many object-type values correspond to numerical data (elements in oil composition).
* **"<1" object:** The object "<1" represents values less than 1.
* **Transformation:** For the project, "<1" will be replaced with 0.
* **Null values:** Some columns are entirely empty, while others contain partial data, as seen in the Non-null column..
""")

# Function to replace "<1" with 0 in the entire DataFrame
def replace_less_than_1(df):
    # Loop through each column in the DataFrame
    for column in df.columns:
        # Check if the column contains strings (to handle cases with "<1")
        if df[column].dtype == object:
            # Replace the "<1" string with the integer 0
            df[column] = df[column].replace('<1', 0)
    return df

# Call the function to modify the DataFrame
df_modified = replace_less_than_1(df)

# Function to convert object columns to numeric only if all values can be converted without NaN
def convert_object_columns_to_numeric(df):
    # Loop through each column in the DataFrame
    for column in df.columns:
        # Check if the column type is 'object'
        if df[column].dtype == object:
            # Try converting the column to numeric using pd.to_numeric
            # errors='coerce' will convert non-numeric values to NaN
            converted_column = pd.to_numeric(df[column], errors='coerce')

            # If the converted column has no NaN values, then convert it to integer
            if not converted_column.isna().any():
                df[column] = converted_column.astype(int)

    return df
# Call the function to modify the DataFrame
df_converted = convert_object_columns_to_numeric(df_modified)

#Detecting Missing Values
st.subheader("Detecting Missing Values")

# Plot the matrix of missing values using missingno
st.write("Matrix of Missing Values:")

plt.figure(figsize=(15, 6))
msno.matrix(df_converted)

# Change the size of labels
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)

# Show in Streamlit
st.pyplot(plt)

st.markdown("""Fully filled columns with a dark color mean 
                there are no missing values. Where white spaces 
                indicate that there are missing values in that column.""")

st.markdown("""
Upon analyzing the chart, it is evident that not all columns are fully populated, with significant gaps present.

Main Key points:
* Specifically, columns such as 'BO3', 'PO4', and 'F' exhibit missing values.
* Additionally, all columns prefixed with 'X' display approximately **16%** missing data.
""")

def replace_missing_with_mean(df):

    for column in df.columns:
        # Check if column is of type 'object' and has 68 missing values
        if df[column].dtype == 'object' and df[column].isna().sum() == 68:
            try:
                # Convert column to numeric (this will raise an error if conversion fails)
                df[column] = pd.to_numeric(df[column], errors='coerce')

                # Calculate the mean of the column, ignoring NaN values
                median_value = df[column].median()

                # Replace NaN values with the mean
                df[column].fillna(median_value, inplace=True)
                print(f"Replaced NaN values in column '{column}' with the mean: {median_value}")

            except ValueError:
                print(f"Could not convert column '{column}' to numeric. Skipping.")

    return df

df_cleaned = replace_missing_with_mean(df_converted)

#Descriptive Statistics
st.subheader("Descriptive Statistics")
st.markdown("""
Main Key points:
* **Central tendency** Calculate mean.
* **Dispersion:** Examine range, and standard deviation.
* **Distribution shape:** **Histograms** for continuous variables and **Box plots** for identify outliers and compare distributions
""")

st.dataframe(df_cleaned.describe())

continuous_vars =  [
     'compartid',
     'oilhours', 'machinehours', 'PQI', 'Fe', 'Cu', 'Cr', 'Pb',
       'Sn', 'Ni', 'Al', 'Si', 'Na', 'K', 'Mo', 'B', 'Ba', 'Mg', 'Ca', 'Zn',
       'P', 'Ag', 'Mn', 'V', 'Ti', 'Cd', 'H2O', 'V40',
       'OXI', 'NIT', 'SUL', 'ISO6', 'ISO14', 'X6', 'X10', 'X14', 'X21', 'X25',
       'X38', 'X70'
    ]

#Histograms
st.subheader("Analysis of Variable Distribution")

# Checkbox to toggle the chart visibility
show_chart_hist = st.checkbox("Show histograms")


if show_chart_hist:
    fig, axes = plt.subplots(10, 4, figsize=(18, 30))

    for i, el in enumerate(continuous_vars):
        ax = axes.flatten()[i]
        df_cleaned[el].plot(kind='hist', bins=30, ax=ax, fontsize='large', color='#fdd39b')
        ax.set_title(el)

    plt.tight_layout()
    st.pyplot(fig)

#Box plots

# Checkbox to toggle the chart visibility
show_chart_box = st.checkbox("Show Box plots")

if show_chart_box:
    fig, axes = plt.subplots(10,4) # create figure and axes

    for i, el in enumerate(list(df_cleaned[continuous_vars].columns.values)):
        a = df_cleaned.boxplot(el, ax=axes.flatten()[i], fontsize='large', color='#d55233')

    fig.set_size_inches(18.5, 30)
    plt.tight_layout()
    st.pyplot(fig)

#Relationship
st.subheader("Relationship")

# Checkbox to toggle the chart visibility
show_chart_corr = st.checkbox("Show heatmap for the correlation matrix")

# Keep only numeric columns
df_numeric = df_cleaned.select_dtypes(include=['number'])

if show_chart_corr:
    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr()

    # Create a Plotly heatmap for the correlation matrix
    fig = px.imshow(corr_matrix,
                    labels=dict(x="Variables", y="Variables", color="Correlation"),
                    x=corr_matrix.columns,  # List of all your 40 variable names in columns
                    y=corr_matrix.columns,  # List of all your 40 variable names in rows
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)

    # Customize layout to fit large number of variables
    fig.update_layout(
        title='Correlation Matrix Heatmap',
        xaxis_tickangle=-45,  # Tilt x-axis labels for better readability
        height=1000,  # Increase the figure height to accommodate all variables
        width=1000,  # Increase the width if necessary
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

#Remove variables that do not contribute anything to the model
st.subheader("Remove variables")
# Dropping the columns
df_cleaned = df_cleaned.drop(columns=['oilgradeid','sampledate','BO3','PO4','Sn','F'])

st.markdown("""
Based on the knowledge gained from data analysis, we apply 
initial variable reduction by removing the following features: 'oilgradeid', 'sampledate', 'BO3', 'PO4', 'Sn', and 'F'.""")

#Build Classification Model
st.subheader("Build Random Forest Classification Model")
st.markdown("""
The main principles of Random Forest:
* **Ensemble Method:** Combines multiple decision trees to improve prediction accuracy and reduce overfitting.
* **Bootstrap Aggregating (Bagging):** Trains each tree on a random subset of the data, ensuring diversity among the trees.
* **Feature Randomness:** At each split in the trees, a random subset of features is considered, enhancing the model's robustness.
* **High Dimensionality:** Effective for datasets with many features, making it suitable for complex problems.
* **No Need for Scaling:** Variables do not need to be scaled or normalized, simplifying preprocessing.
* **Versatile Applications:** Useful in various fields, including healthcare, finance, and marketing, for both classification and regression tasks.
""")

st.markdown("""In the context of machine learning with the Random Forest model, we perform coding for the 
                target column. Below is the column coding for the target values: (A, B, C, X → 0, 1, 2, 3).""")

# Initialization LabelEncoder
le = LabelEncoder()

# Column coding target (A, B, C, X -> 0, 1, 2, 3)
df_cleaned['evalcode_encoded'] = le.fit_transform(df_cleaned['evalcode'])
df_cleaned=df_cleaned.drop(columns=['evalcode'])
X=df_cleaned.drop(['evalcode_encoded'],axis=1)
X=X.join(pd.get_dummies(X.oiltypeid)).drop(['oiltypeid'], axis=1)
y=df_cleaned['evalcode_encoded']

#Tranformation for model
X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)

#Train data
train_data=X_train.join(y_train)
X_train=train_data.drop(['evalcode_encoded'],axis=1)

X_train_np = X_train.to_numpy()
y_train= y_train.to_numpy()

#Test data
test_data=X_test.join(y_test)
X_test=test_data.drop(['evalcode_encoded'],axis=1)

X_test_np = X_test.to_numpy()
y_test = y_test.to_numpy()

#RandomForest Model check max Depth vs. Accuracy
depths = range(1, 10)
scores = []

# Calculating Average Accuracy for Different Depths
for depth in depths:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    score = cross_val_score(model, X, y, cv=5)  # 5-krotna walidacja krzyżowa
    scores.append(np.mean(score))

show_chart_depths = st.checkbox("Show the chart for showing the average accuracy for different depths")

if show_chart_depths:

    st.title('Max Depth vs. Accuracy')
    plt.figure(figsize=(10, 6))
    plt.plot(depths, scores, marker='o',color='#d55233')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Max Depth vs. Accuracy')
    plt.xticks(depths)
    plt.grid()

    st.pyplot(plt)

best_depth = 7
model = RandomForestClassifier(max_depth=best_depth, random_state=42)

# Train model
model.fit(X_train_np, y_train)
y_pred = model.predict(X_test_np)

#Results
st.subheader("Results of Random Forest Model")
# Accuracy
accuracy = round(accuracy_score(y_test, y_pred),2)
st.write('Accuracy on test set: ' + str(accuracy)+ '.')

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.subheader("Enhancing Random Forest Classification with SMOTE Data Augmentation")
st.markdown("""
Using SMOTE for oversampling is an effective way to tackle class imbalance, 
especially in scenarios where minority classes are critical to the problem being solved. 
By generating synthetic data points, SMOTE helps improve model accuracy and robustness, 
leading to better overall performance.

Key Points:
* **Class Imbalance:** 'X' class in the dataset has very few samples.
* **Impact on Model:** Low representation can lead to biased predictions, affecting accuracy and recall for the minority class.
* **Solution:** Apply SMOTE to generate synthetic samples for the underrepresented class.
* **Benefit:** Improves model learning and performance across all classes, leading to more balanced and reliable predictions.
""")

smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
X_train_np, X_test_np, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model_smote = RandomForestClassifier(max_depth=7, random_state=42)

# Train SMOTE
model_smote.fit(X_train_np, y_train)
y_pred = model_smote.predict(X_test_np)

# Accuracy
accuracy_smote = round(accuracy_score(y_test, y_pred),2)
st.write('Accuracy on test set: ' + str(accuracy_smote)+ '.')

st.subheader("Classification Report for Random Forest Classification with SMOTE Data Augmentation")
report_smote = classification_report(y_test, y_pred, output_dict=True)
report_df_smote = pd.DataFrame(report_smote).transpose()
st.dataframe(report_df_smote)


st.markdown("""
Key Point:
*   The use of SMOTE for oversampling has significantly improved our model's performance. By balancing the dataset, 
        we enhanced classification accuracy and reduced class imbalance effects, 
        leading to a more robust and reliable predictive capability.""")

st.subheader("Random Forest with Feature Selection")
# Feature Selection with Threshold 0.03
selector = SelectFromModel(model, threshold=0.03)
X_train_r = selector.transform(X_train_np)

st.write(f"Number of Features After Reduction: {X_train_r.shape[1]}")
X_test_reduced = selector.transform(X_test_np)

# Train model with reduced variables
model_smote.fit(X_train_r, y_train)
y_pred = model_smote.predict(X_test_reduced)
accuracy_r = round(accuracy_score(y_test, y_pred),2)
st.write('Accuracy on test set: ' + str(accuracy_r)+ '.')

st.subheader("Classification Report for Random Forest Classification with Feature Selection")
report_r = classification_report(y_test, y_pred, output_dict=True)
report_r_df_smote = pd.DataFrame(report_r).transpose()
st.dataframe(report_r_df_smote)

important_features = selector.get_support()
selected_feature_names = X_train.columns[important_features]
selected_feature_names_list = list(selected_feature_names)
st.write(f"Selected features are: {selected_feature_names_list}")

st.markdown("""
Key Point:
*   By minimizing the number of features, we created a more efficient model that remains effective 
    but is easier to interpret and deploy. This approach not only streamlines 
    the analysis but also enhances the model's usability in practical applications..""")

# Use the selected feature names to create a new DataFrame from X_train
X_train_reduced_df = X_train[selected_feature_names]

st.subheader("Testing the Random Forest Model with Sample Data")
# Test
#selected_vars = st.multiselect("Select variables:", list(variable_ranges.keys()), default=list(variable_ranges.keys())[:5])

st.write("Please click on 'Predict' to view the oil type result you will receive. "
         "This outcome is based on a model that has undergone variable reduction using specific data parameters. "
         "The data for which the model predicts is provided in the table below.")

#Ranges for variables
variable_ranges = {col: (X_train_reduced_df[col].min(), X_train_reduced_df[col].max()) for col in X_train_reduced_df.columns}

# Slider tylko dla PQI
# PQI_min, PQI_max = variable_ranges['PQI']
# PQI_value = st.slider('Select value fora PQI',
#                                 min_value=int(PQI_min),
#                                 max_value=int(PQI_max),
#                                 value=round((PQI_min + PQI_max) / 2.0))


input_data = {}
for var in variable_ranges.keys():
        input_data[var] = round(np.random.uniform(variable_ranges[var][0], variable_ranges[var][1]))

# Prediction
if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    prediction = model_smote.predict(input_df)
    original_labels = le.inverse_transform(prediction)
    st.dataframe(input_df)
    st.write(f'Predicted type of oil: {original_labels[0]}')