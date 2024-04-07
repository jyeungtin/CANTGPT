# Prompt template

Here you will find three sets of data:

1. `CV_data_{can/en}`: CV data used in the prompt control experiment, in Cantonese or English
2. `Prompt_control_{can/en}`: Prompt templates for prompt control and further experiment, in Cantonese or English
3. `CV_variables_{can/en}`: Countefactual attributes for persona generation, in Cantonese or English

## How to generate the personas
You will need `CV_variables_can` or `CV_variables_en` to generate the personas of any combination of attributes. The method to generate these personas is covered in a separate notebook.

```Python
from itertools import product

# load CV_variables

CV_variables = pd.read_csv('../Data/Prompt template/CV_variables.csv')
CV_variables_can = pd.read_csv('../Data/Prompt template/CV_variables_can.csv')

# grouping variables

grouped_by_var = CV_variables.groupby('Var')['Content'].apply(list)
combinations = list(product(*grouped_by_var)) # cartesan product: generate all possible 3^9 combinations

grouped_by_var_can = CV_variables_can.groupby('Var')['Content'].apply(list)
combinations_can = list(product(*grouped_by_var_can))

# to df
all_combinations_df = pd.DataFrame(combinations, columns=grouped_by_var.index)
all_combinations_df_can = pd.DataFrame(combinations_can, columns=grouped_by_var_can.index)

# these two should correspond to each other
display(all_combinations_df.head(2))
display(all_combinations_df_can.head(2))
```


## Note on prompt control
I have manually written variations of the prompt templates to control for potential variability in ChatGPT's response. Note that only the `SOP` category was used in the study, although you may want to experiment with candidate ranking using the templates I made. 
