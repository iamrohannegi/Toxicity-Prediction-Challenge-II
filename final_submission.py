import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


#########################
###  Preprocessing    ###
#########################

# Train and Test files' relative path
toxicity_file = 'train.csv'
test_file = 'test.csv'

toxicity_data = pd.read_csv(toxicity_file)
test_data = pd.read_csv(test_file, names=['Id'])
test_data = test_data.iloc[1:]

# Extract assay_id and chemical_id from the Id column of the dataframes
toxicity_data['assay_id'] = toxicity_data['Id'].str.split(';').str[1]
toxicity_data['assay_id'] = toxicity_data['assay_id'].map(lambda val: int(val))
toxicity_data['chemical_id'] = toxicity_data['Id'].str.split(';').str[0]

test_data['assay_id'] = test_data['Id'].str.split(';').str[1]
test_data['assay_id'] = test_data['assay_id'].map(lambda val: int(val))
test_data['chemical_id'] = test_data['Id'].str.split(';').str[0]

# Remove rows with Si
toxicity_data = toxicity_data.loc[toxicity_data['chemical_id'] != 'F[Si-2](F)(F)(F)(F)F.[Na+].[Na+]']

# 2D RDKit Descriptors selected
desc_list = ['Chi2v', 'fr_phenol_noOrthoHbond', 'VSA_EState10', 'Chi1n', 'VSA_EState3', 'SlogP_VSA12', 'Kappa2', 'fr_ketone_Topliss', 'SMR_VSA4', 'BCUT2D_MRHI', 'NumSaturatedCarbocycles', 'EState_VSA3', 'NumRadicalElectrons', 'Chi1', 'fr_NH0', 'fr_aryl_methyl', 'fr_para_hydroxylation', 'Chi3v', 'fr_diazo', 'VSA_EState7', 'fr_Ar_NH', 'fr_oxazole', 'fr_alkyl_carbamate', 'fr_Ndealkylation1', 'SlogP_VSA10', 'TPSA', 'SlogP_VSA5', 'NumAliphaticCarbocycles', 'PEOE_VSA2', 'fr_C_O', 'fr_C_S', 'fr_guanido', 'NumHAcceptors', 'MolLogP', 'VSA_EState2', 'fr_lactone', 'PEOE_VSA9', 'Chi0', 'fr_term_acetylene', 'BCUT2D_CHGHI', 'Kappa1', 'fr_Al_OH_noTert', 'MaxEStateIndex', 'fr_phenol', 'MaxPartialCharge', 'Chi0n', 'NumAliphaticHeterocycles', 'EState_VSA4', 'EState_VSA10', 'fr_ether', 'fr_Al_OH', 'fr_piperzine', 'fr_ester', 'fr_thiocyan', 'fr_isocyan', 'fr_COO2', 'FpDensityMorgan3', 'fr_unbrch_alkane', 'PEOE_VSA8', 'fr_azide', 'NumAromaticCarbocycles', 'SMR_VSA1', 'VSA_EState9', 'fr_phos_acid', 'PEOE_VSA4', 'BCUT2D_MWLOW', 'Chi0v', 'fr_NH2', 'fr_nitro_arom', 'BalabanJ', 'MaxAbsPartialCharge', 'LabuteASA', 'fr_C_O_noCOO', 'fr_Ar_N', 'fr_N_O', 'fr_Nhpyrrole', 'fr_hdrzone', 'fr_imidazole', 'fr_tetrazole', 'NumAromaticHeterocycles', 'fr_thiophene', 'fr_allylic_oxid', 'NumSaturatedHeterocycles', 'SlogP_VSA6', 'fr_sulfonamd', 'NHOHCount', 'PEOE_VSA12', 'PEOE_VSA6', 'Kappa3', 'PEOE_VSA14', 'fr_azo', 'fr_quatN', 'fr_epoxide', 'NumHeteroatoms', 'MinAbsPartialCharge', 'FpDensityMorgan2', 'fr_ketone', 'HeavyAtomCount', 'Chi1v', 'SMR_VSA3', 'PEOE_VSA1', 'fr_SH', 'fr_nitro_arom_nonortho', 'fr_Ndealkylation2', 'fr_phos_ester', 'fr_oxime', 'BCUT2D_MRLOW', 'MaxAbsEStateIndex', 'fr_Al_COO', 'fr_priamide', 'fr_piperdine', 'SMR_VSA5', 'Chi4v', 'SlogP_VSA4', 'fr_pyridine', 'SMR_VSA6', 'SlogP_VSA7', 'NumHDonors', 'EState_VSA2', 'VSA_EState5', 'ExactMolWt', 'PEOE_VSA11', 'EState_VSA9', 'fr_halogen', 'VSA_EState6', 'Chi2n', 'PEOE_VSA7', 'fr_COO', 'NOCount', 'fr_hdrzine', 'NumAliphaticRings', 'fr_sulfide', 'SlogP_VSA8', 'BCUT2D_LOGPHI', 'fr_barbitur', 'fr_isothiocyan', 'MolWt', 'MinAbsEStateIndex', 'fr_furan', 'BCUT2D_CHGLO', 'SlogP_VSA9', 'Chi4n', 'fr_urea', 'MolMR', 'fr_morpholine', 'SMR_VSA10', 'NumRotatableBonds', 'fr_ArN', 'fr_thiazole', 'SlogP_VSA11', 'PEOE_VSA5', 'SMR_VSA8', 'VSA_EState8', 'HeavyAtomMolWt', 'BCUT2D_MWHI', 'NumValenceElectrons', 'fr_nitro', 'fr_benzodiazepine', 'EState_VSA11', 'Ipc', 'NumSaturatedRings', 'fr_benzene', 'PEOE_VSA10', 'VSA_EState1', 'FpDensityMorgan1', 'fr_aldehyde', 'SMR_VSA7', 'HallKierAlpha', 'fr_Imine', 'fr_alkyl_halide', 'SlogP_VSA3', 'fr_amidine', 'EState_VSA6', 'fr_imide', 'fr_bicyclic', 'RingCount', 'fr_lactam', 'SlogP_VSA2', 'MinPartialCharge', 'Chi3n', 'EState_VSA7', 'FractionCSP3', 'SMR_VSA9', 'EState_VSA8', 'BertzCT', 'qed', 'NumAromaticRings', 'fr_prisulfonamd', 'SMR_VSA2', 'fr_NH1', 'fr_Ar_COO', 'EState_VSA5', 'fr_HOCCN', 'PEOE_VSA13', 'fr_amide', 'PEOE_VSA3', 'MinEStateIndex', 'fr_Ar_OH', 'VSA_EState4', 'SlogP_VSA1', 'fr_sulfone', 'BCUT2D_LOGPLOW', 'EState_VSA1', 'fr_nitrile', 'fr_nitroso']

# Now we need to generate the values for the above RDKit descriptor for each row
# First we reindex the data frames and make sure that each row has a column for 
# the descriptors above.
toxicity_data = toxicity_data.reindex(columns=(list(toxicity_data.columns) + desc_list))
test_data = test_data.reindex(columns=(list(test_data.columns) + desc_list))

# Utility function that takes a row of data frame and updates it 
# with all the value of above selected RDKit descriptors based on the
# chemical id of that row.  
def apply_mol_desc(row):
    mol = Chem.MolFromSmiles(row.chemical_id)
    for desc in desc_list:
        row[desc] = getattr(Descriptors, desc)(mol)
    return row

# To get the final data frame with all RDKit Descriptors value, we call our utility function
final_toxicity_data = toxicity_data.apply(apply_mol_desc, axis='columns')
test_data = test_data.apply(apply_mol_desc, axis='columns')


#########################
## Feature Selection   ##
#########################

# Our features will be the assay_id and 2D RDKit descriptors selected above
features = ['assay_id'] + desc_list


# Get the X and y to train the model
X = final_toxicity_data[features]
y = final_toxicity_data['Expected']

# This version of XGBModel only works with 0s and 1s y value, our data contains 1s and 2s
# So we need to encode them using LabelEncoder
le = LabelEncoder()

#########################
## Testing the model   ##
#########################

# Break train dataset to 80-20 parts for internal evaluation
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0, test_size=0.2)
# y_train = le.fit_transform(y_train)

# xgb_model =XGBClassifier(n_estimators=3000, learning_rate=0.09  , early_stopping_rounds=7, random_state=0)
# xgb_model.fit(X_train, y_train,
#             eval_set=[(X_train, y_train)],
#             verbose=False)

# y_pred = xgb_model.predict(X_test)

# Inverse transform the predictions to get the result back to 1s and 2s
# instead of 0s and 1s using the label encoder
# y_pred = le.inverse_transform(y_pred)

# f1 = f1_score(y_test, y_pred)

# print('F1 Score: ', f1)

################################
## Training with entire data  ##
################################

y = le.fit_transform(y)

xgb_model = XGBClassifier(n_estimators=3000, learning_rate=0.09, early_stopping_rounds=7, random_state=0)
xgb_model.fit(X, y,
            eval_set=[(X, y)],
            verbose=False)


#########################
###  Predictions      ###
#########################
X_test = test_data[features]
y_pred = xgb_model.predict(X_test)

y_pred = le.inverse_transform(y_pred)

# Finally, produce a submission.csv file with our predictions
output = pd.DataFrame({'Id': test_data.Id,
                       'Predicted': y_pred})
output.to_csv('submission.csv', index=False)