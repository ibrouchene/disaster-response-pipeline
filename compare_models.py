import os
import pandas as pd
import numpy as np


def highlight_max(s):
    s = s[1:]
    if s.dtype == np.object:
        is_max = [False] + [False for _ in range(s.shape[0])]
    else:
        is_max = s == s.max()
    return ['background: lightgreen' if cell else '' for cell in is_max]


if __name__ == '__main__':
    model_results = r".\model_results"
    merged_dataframes = {"precision": pd.DataFrame(), "recall": pd.DataFrame(), "f1-score": pd.DataFrame(),
                         "support": pd.DataFrame()
                         }
    for model, _, _ in os.walk(model_results):
        if model != model_results:
            current_model_string = model.split('\\')[-1]
            res = pd.read_csv(os.path.join(model, "model_results.csv"), sep=";")
            res.columns = [str(col) + current_model_string[-2:] if idx != 0 else col for idx, col in
                           enumerate(res.columns)]
            all_models = []
            for metric in ["precision", "recall", "f1-score", "support"]:
                # Get corresponding container
                df = merged_dataframes[metric]
                # Cut only precision part
                res_reduced = res[["Unnamed: 0", metric + current_model_string[-2:]]]
                if df.size == 0:
                    df = res_reduced
                else:
                    df = df.merge(res_reduced, how='right')
                previous_model_string = current_model_string
                merged_dataframes[metric] = df
                all_models.append(previous_model_string)

    cleaned = merged_dataframe.drop(all_models[1:], axis=1)
    tmp = cleaned.T
    tmp.style.apply(highlight_max)
    print("done")