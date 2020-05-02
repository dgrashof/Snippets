from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def df_chi_square(df,fields,response,iterations):
    def chi_square(val_1,val_2,prob):
        table = pd.crosstab(val_1,val_2)
        stat, p, dof, expected = chi2_contingency(table)
        prob = prob
        return(p)
    val = iterations
    pair_df = pd.DataFrame()
    for i in fields:
        j = 0
        p_values = []
        while j<val:
            df = df
            df1 = df[df[response]==1]
            df2 = df[df[response]==0].sample(len(df1))
            df_test = pd.concat([df1,df2])
            p_values.append(chi_square(val_1 = df_test[i],val_2 = df_test[response],prob = .95))
            j = j+1
        pair = {'field':i,'value':np.mean(p_values)}
        pair_df = pair_df.append(pair,ignore_index = True)
    return(pair_df)