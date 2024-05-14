from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

available_datasets = [
    'dfdc-35-5-10',
    'ff-c23-720-140-140',
    'ff-c23-720-140-140-5fpv',
    'ff-c23-720-140-140-10fpv',
    'ff-c23-720-140-140-15fpv',
    'ff-c23-720-140-140-20fpv',
    'ff-c23-720-140-140-25fpv',
    'celebdf', 
]


def load_df(celeb_df_path: str, celeb_faces_dir: str, dataset: str) -> (pd.DataFrame, str):

    if dataset.startswith('celeb'):
        df = pd.read_pickle(celeb_df_path)
        root = celeb_faces_dir    
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))
    return df, root


def get_split_df(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:

    if dataset == 'celebdf':

        seed = 41
        num_real_train = 600

        # Save random state
        st0 = np.random.get_state()
        # Set seed for this selection only
        np.random.seed(seed)
        # print("1")
        
        # print((df['class'] == 'Celeb-synthesis').any())
        # df = df.rename_axis('facepath')
        # print(df['facepath'])


        # Read the text file and extract video paths
        with open('E:/MastersProject/fakedetector-main/fakedetector-main/faces_output/List_of_testing_videos.txt', 'r') as file:
            video_paths = [line.split()[1].split(".mp4")[0] for line in file]
        # Replace all occurrences of '/' with '\' in each video path
        video_paths = [path.replace('/', '\\') for path in video_paths]

        # print(video_paths)
        # print(df.index)

        # Create a new column initialized with False
        df['test'] = False

        # Iterate over each index (facepath) in the DataFrame
        for facepath in df.index:
            # Check if the video path is in the list of video paths
            if facepath.split('.mp4')[0] in video_paths:
                # If the video path matches, set the corresponding value in the new column to True
                df.at[facepath, 'test'] = True

        # Print the updated DataFrame
        # print(df.columns)
        # print(df.tail(10))
        # # Filter rows where the index contains the substring "YouTube-real/00170"
        # print((df['test'] == True).sum()) 16565

        # Split on original videos
        random_train_val_real_videos = np.random.permutation(
            df[(df['label'] == False) & (df['test'] == False)]['video'].unique())
        # print("2")
        train_orig = random_train_val_real_videos[:num_real_train]
        val_orig = random_train_val_real_videos[num_real_train:]
        if split == 'train':
            split_df = pd.concat((df[df['original'].isin(train_orig)], df[df['video'].isin(train_orig)]), axis=0)
            # print("train data")
        elif split == 'val':
            split_df = pd.concat((df[df['original'].isin(val_orig)], df[df['video'].isin(val_orig)]), axis=0)
            # print("val data")
        elif split == 'test':
            split_df = df[df['test'] == True]
            # print("test data")
        else:
            raise NotImplementedError('Unknown split: {}'.format(split))
        # Restore random state
        np.random.set_state(st0)


    
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))
    return split_df


def make_splits(celeb_df: str, celeb_dir:str, dbs: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple[pd.DataFrame, str]]]:

    split_dict = {}
    full_dfs = {}
    for split_name, split_dbs in dbs.items():
        split_dict[split_name] = dict()
        for split_db in split_dbs:
            if split_db not in full_dfs:
                full_dfs[split_db] = load_df(celeb_df,celeb_dir, split_db)
            full_df, root = full_dfs[split_db]
            split_df = get_split_df(df=full_df, dataset=split_db, split=split_name)
            split_dict[split_name][split_db] = (split_df, root)

    return split_dict
