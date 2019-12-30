def clean_data(df, row_proportion_thresh, col_proportion_thresh, outlier_threshold):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    df_feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep = ';')
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    X_labeled_missing_values = df_feat_info[df_feat_info.missing_or_unknown.isin(['[-1,X]','[XX]', '[-1,XX]'])]
    x_labeld_list = X_labeled_missing_values.attribute.tolist()
    
    for column in df.columns:
        feature = df_feat_info[df_feat_info['attribute'] == column].iloc[0]
        feature_unknow_label = feature['missing_or_unknown']
        if feature['attribute'] in x_labeld_list:
            x_str = feature_unknow_label[1:-1].split(',')
            missing_labels = [int(value) if 'X' not in value else value for value in x_str]
        else:
            missing_labels = ast.literal_eval(feature_unknow_label)
        for label in missing_labels:
            df[column].replace(label, np.NaN, inplace = True) 
  
    # remove selected columns and rows, ...
    column_null_list = df.isnull().sum()
    column_null_proportion = column_null_list / len(df) * 100
    column_outliers = column_null_proportion[column_null_proportion > col_proportion_thresh]
    df.drop(column_outliers.index, axis=1, inplace=True) 
    print('outlier columns removed: \n%s' % column_outliers)
    row_null_list = df.isnull().sum(axis=1)
    row_null_proportion = row_null_list / len(df.columns) * 100
    rows_outliers = row_null_proportion[row_null_proportion > row_proportion_thresh]
    df.drop(rows_outliers.index, axis=0, inplace=True) 
    
    # removing outliers
    df = mask_outliers(df, outlier_threshold)

    # select, re-encode, and engineer column values.
    categorical_features_list = df_feat_info[df_feat_info["type"]=="categorical"].attribute.tolist()
    categorical_features_list_in_df = [x for x in categorical_features_list if x in df.columns.tolist()]
    categorical_series = df[categorical_features_list_in_df].nunique().sort_values()
    boolean_not_binary_series = categorical_series > 2 
    not_binary_features_list = boolean_not_binary_series[boolean_not_binary_series].index.tolist()
    df = pd.get_dummies(df, columns = not_binary_features_list)    
    df.OST_WEST_KZ.replace({'O': 0, 'W': 1}, inplace=True)
        
    # One-hot encode 
    df = features_from_PRAEGENDE_JUGENDJAHRE(df)
    df = features_from_CAMEO_INTL_2015(df)
    df = pd.get_dummies(df, columns = ['DECADE', 'MOVEMENT'])
    df.drop('PRAEGENDE_JUGENDJAHRE', axis = 1, inplace=True)
    df = pd.get_dummies(df, columns = ['WEALTH', 'LIFE_STAGE'])
    df.drop('CAMEO_INTL_2015', axis = 1, inplace=True)

    # Return the cleaned dataframe.
    
    return df

def mask_outliers(df, prop_threshold):
    """
    Mask the outliers as nans.
    """
    outliers_replaced = []
    df_columns = df.columns.tolist()
    total_records = df.shape[0]
    
    for c in df_columns:
        nulls = df[c].isnull().sum()
        non_nulls = total_records - nulls
        element_proportion_serie = df.groupby(c)[c].count()/non_nulls*100

        element_list = element_proportion_serie.keys().tolist()
        proportion_list = element_proportion_serie.values.tolist()

        # masking values whom proportion is below a threshold
        outlier_elements = [e for e in element_list if element_proportion_serie[e] < prop_threshold]
        for outlier in outlier_elements:
            outliers_replaced.append('outlier: %s, proportion: %s ,column: %s' %(outlier, element_proportion_serie[outlier], c))
            df[c].replace(outlier, np.NaN, inplace = True)

    return df

def features_from_PRAEGENDE_JUGENDJAHRE(df):
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([1,2]), 'DECADE'] = '40s'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([3,4]), 'DECADE'] = '50s'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([5,6,7]), 'DECADE'] = '60s'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([8,9]), 'DECADE'] = '70s'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([10,11,12,13]), 'DECADE'] = '80s'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([14,15]), 'DECADE'] = '90s'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([1,3,5,8,10,12,14]), 'MOVEMENT'] = 'Mainstream'
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([2,46,7,9,11,13,15]), 'MOVEMENT'] = 'Avantgarde'
    return df


def features_from_CAMEO_INTL_2015(df):
    df["CAMEO_INTL_2015"] = df["CAMEO_INTL_2015"].astype(float)

    df.loc[df["CAMEO_INTL_2015"] // 10 == 1.0, 'WEALTH'] = 'Wealthy Households'
    df.loc[df["CAMEO_INTL_2015"] // 10 == 2.0, 'WEALTH'] = 'Prosperous Households'
    df.loc[df["CAMEO_INTL_2015"] // 10 == 3.0, 'WEALTH'] = 'Comfortable Households'
    df.loc[df["CAMEO_INTL_2015"] // 10 == 4.0, 'WEALTH'] = 'Less Affluent Households'
    df.loc[df["CAMEO_INTL_2015"] // 10 == 5.0, 'WEALTH'] = 'Poorer Households'
    df.loc[df["CAMEO_INTL_2015"] % 10 == 1.0, 'LIFE_STAGE'] = 'Pre-Family Couples & Singles'
    df.loc[df["CAMEO_INTL_2015"] % 10 == 2.0, 'LIFE_STAGE'] = 'Young Couples With Children'
    df.loc[df["CAMEO_INTL_2015"] % 10 == 3.0, 'LIFE_STAGE'] = 'Families With School Age Children'
    df.loc[df["CAMEO_INTL_2015"] % 10 == 4.0, 'LIFE_STAGE'] = 'Older Families &  Mature Couples'
    df.loc[df["CAMEO_INTL_2015"] % 10 == 5.0, 'LIFE_STAGE'] = 'Elders In Retirement'
    return df