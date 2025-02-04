# Ryan Wolstenholme (yyp6zx)


# Question 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### 1. Clean Price Variable in Airbnb Data ###
airbnb_df = pd.read_csv("airbnb_hw.csv")

# Remove dollar signs and commas, then convert to numeric
airbnb_df['Price'] = airbnb_df['Price'].astype(str).str.replace(r'[\$,]', '', regex=True)
airbnb_df['Price'] = pd.to_numeric(airbnb_df['Price'], errors='coerce')

# Count missing values
missing_price = airbnb_df['Price'].isna().sum()
print(f"Missing values in Price: {missing_price}")

### 2. Clean subject_injury in MN Police Use-of-Force Data ###
police_df = pd.read_csv("mn_police_use_of_force.csv")

# Replace missing values with "No" and standardize case
police_df['subject_injury'] = police_df['subject_injury'].fillna("No").str.strip().str.capitalize()

# Compute proportion of missing values before cleaning
missing_subject_injury = police_df['subject_injury'].isna().mean()
print(f"Proportion of missing values in subject_injury (before cleaning): {missing_subject_injury:.2%}")

# Cross-tabulation of subject_injury with force_type
force_crosstab = pd.crosstab(police_df['subject_injury'], police_df['force_type'])
print(force_crosstab)



### 3. Clean WhetherDefendantWasReleasedPretrial Variable ###
# Load the justice data file
justice_df = pd.read_parquet("justice_data.parquet")

# Check if the column exists
if 'WhetherDefendantWasReleasedPretrial' in justice_df.columns:
    # Convert Yes/No to 1/0, keeping missing values
    justice_df['WhetherDefendantWasReleasedPretrial'] = justice_df['WhetherDefendantWasReleasedPretrial'].replace(
        {"Yes": 1, "No": 0}
    ).astype(float)
else:
    print("Column 'WhetherDefendantWasReleasedPretrial' not found in dataset.")



### 4. Clean ImposedSentenceAllChargeInContactEvent ###
# Check if the column exists
if 'ImposedSentenceAllChargeInContactEvent' in justice_df.columns and 'SentenceTypeAllChargesAtConvictionInContactEvent' in justice_df.columns:
    # Count missing values
    missing_sentence = justice_df['ImposedSentenceAllChargeInContactEvent'].isna().sum()
    print(f"Missing values in ImposedSentenceAllChargeInContactEvent: {missing_sentence}")

    # Compare missingness with SentenceTypeAllChargesAtConvictionInContactEvent
    sentence_type_distribution = justice_df.groupby('SentenceTypeAllChargesAtConvictionInContactEvent')[
        'ImposedSentenceAllChargeInContactEvent'
    ].apply(lambda x: x.isna().mean())

    print(sentence_type_distribution)
else:
    print("Column 'ImposedSentenceAllChargeInContactEvent' or 'SentenceTypeAllChargesAtConvictionInContactEvent' not found.")



# Questions 2

# Load the Excel file
shark_df = pd.read_excel("shark_data.xls")

# Print column names to check for issues
print("Original Column Names:", shark_df.columns)

# Standardize column names
shark_df.columns = shark_df.columns.str.strip()

# Print cleaned column names
print("Cleaned Column Names:", shark_df.columns)


# Drop completely empty columns
shark_df = shark_df.dropna(axis=1, how='all')

# Clean and analyze the Year variable
shark_df['Year'] = pd.to_numeric(shark_df['Year'], errors='coerce')
print("Year Range: {shark_df['Year'].min()} to {shark_df['Year'].max()}")

# Filter for attacks since 1940
shark_recent = shark_df[shark_df['Year'] >= 1940]

# Analyze trends over time
attacks_per_year = shark_recent.groupby('Year').size()
plt.figure(figsize=(12, 6))
plt.plot(attacks_per_year, marker='o')
plt.xlabel("Year")
plt.ylabel("Number of Attacks")
plt.title("Shark Attacks Over Time (Since 1940)")
plt.show()

# Clean and analyze the Age variable
shark_df['Age'] = pd.to_numeric(shark_df['Age'], errors='coerce')
plt.figure(figsize=(8, 5))
plt.hist(shark_df['Age'].dropna(), bins=20, edgecolor='black')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Histogram of Victim Ages")
plt.show()

# Proportion of victims that are male
male_proportion = (shark_df['Sex'] == 'M').mean()
print(f"Proportion of male victims: {male_proportion:.2%}")

# Clean the Type variable
shark_df['Type'] = shark_df['Type'].str.strip().str.capitalize()
shark_df['Type'] = shark_df['Type'].apply(lambda x: x if x in ['Provoked', 'Unprovoked'] else 'Unknown')

# Proportion of attacks that are unprovoked
unprovoked_proportion = (shark_df['Type'] == 'Unprovoked').mean()
print(f"Proportion of unprovoked attacks: {unprovoked_proportion:.2%}")

# Clean the Fatal Y/N variable
shark_df['Fatal Y/N'] = shark_df['Fatal Y/N'].apply(lambda x: x if x in ['Y', 'N'] else 'Unknown')

# Are unprovoked attacks more common on men or women?
attack_gender = pd.crosstab(shark_df['Sex'], shark_df['Type'], normalize='columns')
print(attack_gender)

# Is fatality rate different for provoked vs. unprovoked?
fatality_rate = pd.crosstab(shark_df['Type'], shark_df['Fatal Y/N'], normalize='index')
print(fatality_rate)

# Is fatality rate different for men vs. women?
gender_fatality = pd.crosstab(shark_df['Sex'], shark_df['Fatal Y/N'], normalize='index')
print(gender_fatality)

# Proportion of attacks by white sharks
shark_df['Species'] = shark_df['Species'].astype(str).str.lower()

shark_df['White_Shark'] = shark_df['Species'].str.contains('white shark', na=False)
white_shark_proportion = shark_df['White_Shark'].mean()
print(f"Proportion of attacks by white sharks: {white_shark_proportion:.2%}")



# Question 4

# 1) The US census collected race data using a self Identification method
# 2) We gather this data to understand the demographics of the United States. This data influences policymakers and laws for certain regions. Data quality matters to ensure proper information.
# 3) Expanded write ins helped people better describe their race/ethnicity. A better description for the difference between race and ethnicity was missing.
#    A better description of races and more options could help represent the diversity within racial groups. Yes some of the census's better aspects could be implemented to acquire better data.
# 4) The census only asked about sex and left out gender as a question. While the question was simple and direct, it left out gender as a question. In the future it should add gender as a question.
# 5) There are several risks with data cleaning when handling sex, gender, etc. Privacy risks, whereas over detailed data can expose individuals. Missing values can lead to underrepresentation.
#     Some bad characteristics people might adopt would be forcing assumptions such as sex based on name. However some good practices might be transparency in how missing data is handled.
# 6) Some concerns I would have would be accuracy and bias which rely on flawed assumptions, ethical issues, legal risks, and social harm.