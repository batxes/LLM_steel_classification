### Assessment Task: Supplier Data Standardization for Metal Trading

#### Background
Vanilla Steel is at the forefront of revolutionizing the metal trading industry by providing digital solutions that streamline transactions and enhance market liquidity. A critical component of our operation involves integrating supplier data from various formats into our centralized system. This task will evaluate your ability to process, clean, and standardize supplier inventory data from multiple formats into an unified data structure.

#### Objective
Develop a prototype solution to automatically extract and standardize supplier data from different file formats into a unified structure.

All three files located in the `./resources` folder represents different formats of supplier data, reflecting the diverse systems used by suppliers in the steel industry.

#### Task Description

The task involves standardising this diverse data into a unified format that can be used and processed by the application or data warehouse. 

1. **Data Understanding and Exploration**
  - Review the provided datasets to understand the structure and type of data contained in each file.
  - Identify key attributes that need to be extracted and standardized across all datasets.
  - Implement methods to extract text data from the CSV files. 
  - You can use any 3rd party libraries such as Use libraries such as `pandas`, `tabula-py`.

2. **Data Processing and Cleaning**
  - Handle inconsistencies in data formats, such as varying column names, data types, and missing values.
  - Clean the data by addressing any anomalies and ensuring consistency in units and descriptions.

3. **Tokenization and Feature Extraction**
  - Tokenize the extracted text using i.e. `spaCy` or similar NLP libraries.
  - Extract relevant features from tokens (e.g., text, position, context).

4. **Classification**
  - Train a machine learning model to classify tokens into predefined categories such as `material_id`, `material_name`, `quantity`, `unit`, `price_per_unit`, `supplier`, `dimensions`, `weight` (feel free to adjust the list, we care about the quality of the solution rather than quantity)
  - Use pre-trained models (e.g., `BERT`, `spaCy's NER`) or train a custom model.

5. **Post-processing**
  - Aggregate the classified tokens into a structured format.
  - Ensure the final output aligns with the defined schema.

6. **Documentation and Presentation**
  - Prepare a brief report or presentation outlining your approach, findings, challenges and recommendations for further improvement.

#### Data overview

The 3 data files contains detailed information about a metal products from a supplier.
   - The data is categorized by attributes like quality, grade, and physical properties such as thickness and width.
   - Additional columns provide information on any defects or special conditions.
   - The files have different structure to represent different type of ERP systems on the market that provides supplier data in the different formats
   - VanillaSteel is an Europe wide company and we have to support multiple languages that is represented in the `./resources/source3.xlsx` file

#### Interpretation of Data Content

###### Source 1
- **Columns:**
  - **Quality/Choice**: Indicates the quality or choice level of the metal (e.g., "2nd").
  - **Grade**: Specifies the grade of the metal, which could be related to its quality or type (e.g., "C100S").
  - **Finish**: Describes the surface treatment of the metal (e.g., "ungebeizt, nicht geglüht").
  - **Thickness (mm)**: Thickness of the metal in millimeters.
  - **Width (mm)**: Width of the metal in millimeters.
  - **Description**: Describes additional characteristics or defects (e.g., "Sollmasse (Gewicht) unterschritten").
  - **Gross weight (kg)**: The total weight of the metal in kilograms.

###### Source 2
- **Columns:**
  - **Material**: Describes the type and specification of the metal, it is 1 long string value that contains grade, width, lenght, coating, e.g., "DX51D +Z140 Ma-C 1,50 x 1350,00 x 2850,00":
    - DX51D: grade
    - Z140: coating (without +)
    - 1,50x1350,00x2850,00: height, width and length
    - Ma-C: finish
  - **Description**: Provides additional information about the material (e.g., "Material is Oiled").
  - **Article ID**: A unique identifier for the material or product.
  - **Weight**: The weight of the material in kilograms.
  - **Quantity**: The number of units or quantity of the material.

###### Source 3
- **Columns:**
  - **Numéro de**: Number or ID for the record.
  - **Article**: A unique identifier for the material or product.
  - **Matériel Desc#**: Describes the type and specification of the metal, it is 1 long string value that contains grade, width, lenght, coating, e.g. "HDC 0.75x1010 GXES G10/10 MB O":
    - HDC: grade
    - 0.75x1010: width, length
    - GXES: coating
    - MB: finish
  - **Unité**: The unit of measurement (e.g., "KG").
  - **Libre**: The number of units or quantity of the material.


#### Deliverables
- Python script containing your solution.
- Python test script validating your solution.
- The standardized dataset in CSV format.
- Report or presentation summarizing your approach and results.
- Clear instructions how to run your script.
- Plan for the future expansion of the algorithm and the potential most difficult challenges.

#### Evaluation Criteria
- **Technical Proficiency**: Ability to use appropriate tools and libraries for data extraction and classification.
- **Problem-Solving Skills**: Effectiveness of the solution in handling unstructured and inconsistent data.
- **Machine Learning Application**: Ability to train and evaluate a model on the provided datasets.

---

### Next steps
- If you need any additional information or clarification, feel free to reach out to us. We are more than happy to assist. 
- If you are unable to fully complete your work, dont worry. Submit everything you have done to the point and provide us written description what have you achieved, what is still TODO and how would you resolve it (written format or pseudo code).

**Good luck!** We look forward to seeing your innovative solutions.

### Files Provided for the Assessment
- [source1](./resources/source1.xlsx)
- [source2](./resources/source2.xlsx)
- [source3](./resources/source3.xlsx)
