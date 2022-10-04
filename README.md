# CS5228 Project

Link to project plan [here](https://docs.google.com/spreadsheets/d/11c65nRfv9-kqoD8SUnnphL5oPNgihyhzmBO6jFLDbSY/edit#gid=0)    
Link to progress report [here](https://docs.google.com/presentation/d/1H8_7V6NpIQwINXYNRxB-Bp6iFr9os3Uu/edit#slide=id.p12)

# Task 1

## 1. Exploratory Data Analysis
Below are the insights gained from exploratory data analysis
#### Target Variable
#### Size Square Feet
#### Built Year
#### Tenure
#### Number of bedrooms
#### Number of bathrooms
#### Property type
#### Latitude and Longitude
#### Planning Area
#### Furnishings
#### Total number of units

## 2. Dealing with missing values

## 3. Preprocessing
### Transformation
- **Tenure:** Binned into "99-year leasehold", "freehold" and "others".
- 

### New features created
- Mean price of different tenure category
- Number of active years since building has been built
- Remaining lease
- Number of total rooms (bedroom + bathroom)
- Size per room (size_sqft/total_rooms)
- Mean price of different property type
- Distance from nearest MRT
- Number of shopping malls in planning area

### Encoding
- **Planning area**: Target encoding
- **Tenure**: One hot encoding after binning
- **Property type**: One hot encoding
- **Furnishing**: One hot encoding
