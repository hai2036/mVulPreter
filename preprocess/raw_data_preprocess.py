import pandas as pd
import os
import difflib
import pickle

def process_raw_data(filter_csv_path, raw_csv_path): # filter dataset is data where is the vul attribute is 1
    if os.path.exists(filter_csv_path): # Check if the filter data exist
        pd_filter = pd.read_csv(filter_csv_path)
        # pd_filter = pd.read_csv('/home/MSR_data_cleaned_pairs_Test.csv')

    else:
        pd_raw = pd.read_csv(raw_csv_path)
        filter_csv = pd_raw.loc[pd_raw["vul"] == 1] # filter dataset to only include the rows where the column vul is 1
        filter_csv.to_csv(filter_csv_path) # save the filter data frame into a new CSV file
        pd_filter = pd.read_csv(filter_csv_path)
    return pd_filter # Return a dataframe of filter dataset

def label(f_vul, f_novul ,label_dict, outfile):
    diff = list(difflib.unified_diff(f_novul.splitlines(), f_vul.splitlines())) # save the difference of f_vul file and f_novul file
    split_list = [i for i,line in enumerate(diff) if line.startswith("@@")] #  save the indices where each difference chunks start
    split_list.append(len(diff))  # add the length of the diff list into the split list
    i = 0
    for i in range(len(split_list) - 1): # process each diff chunks
        start = split_list[i] # set the staring indices of this chunk
        del_linenum = diff[start].split("@@ -")[-1].split(",")[0].split('+')[-1].strip() # extract the starting line number from diff header
        end = split_list[i + 1] # et the staring indices of next chunk
        
        line_num = int(del_linenum) # set the starting line num
        for line in diff[start+1 : end]: # for each line in this chunk
            if line.startswith("-"): # If this line is removed then we add it into label_dict
                label_dict[outfile].append(line_num)
            elif line.startswith("+"): # If this line is added then we subtract 1 from line_num to balance out
                line_num -= 1
            line_num += 1 # After each line processed we increase line_num by 1
        i += 1 # move to next chunk
        

def main():
    dataset_path = '/content/mVulPreter/dataset/'
    raw_data_path = 'raw_data/'
    raw_data_filename = 'MSR_data_cleaned.csv'
    filter_data_filename = 'MSR_data_filtered.csv'

    dataset_path_output = 'dataset_test/'
    label_pkl_file = 'test_label_pkl.pkl'
    
    filter_csv_path = dataset_path + raw_data_path + filter_data_filename
    raw_csv_path = dataset_path + raw_data_path + raw_data_filename
    output_path = dataset_path + dataset_path_output
    pkl_path = dataset_path + label_pkl_file

    pd_filter = process_raw_data(filter_csv_path, raw_csv_path)
    file_cnt = pd_filter.shape[0] # the total number of rows in filter dataset
    file_num = 0
    label_dict = {}
    cnt_1 = 0
    for index, row in pd_filter.iterrows(): # Loop through each row in the filter dataset
        file_num += 1
        print(str(file_num) + ' / ' + str(file_cnt)) # print out how many row it has done / total row
        project_name = row["project"] # take the value of the project col
        hash_vaule = row['commit_id'] # take the value of the commit_id col
        file_name = project_name + "_" + hash_vaule 
        outfile = output_path + file_name # the output file location: dataset/dataset_test/file_name

        #file_name_cnt = 0
        outfile_new = outfile
        label_key = '1_'+ file_name
        #if label_key in label_dict.keys(): # Check if the output file is in the label_dict
        #    outfile_new = outfile + '_' + str(file_name_cnt)
        #    file_name_cnt += 1


        if not os.path.exists(outfile_new): 
            os.mkdir(outfile_new)

        label_dict[label_key] = [] # add outfile_new into label_dict | key: outfile_new; value: []

        func_before = row['func_before']
        func_after = row["func_after"]
        vul_file_name = '1_'+ file_name + '.c'
        novul_file_name = '0_' + file_name+ '.c'

        with open(outfile_new + '/'+ vul_file_name, 'w', encoding='utf-8') as f_vul: # create a vul file
            f_vul.write(func_before)
            cnt_1 += 1

        with open(outfile_new + '/' + novul_file_name, 'w', encoding='utf-8') as f_novul: # create a no vul file
            f_novul.write(func_after)
            cnt_1 += 1

        if pd.isnull(row['lines_before']):
            label_dict[label_key] = ['']  # add outfile_new into label_dict | key: outfile_new; value: []

        else:
            label(func_before, func_after, label_dict, label_key)

        if file_num == 2: break # limit the ammout of function files

    with open(pkl_path,'wb') as f: 
        pickle.dump(label_dict, f) # dump the label file as test_label_pkl.pkl

    print(cnt_1) # print out the number of file that it has created


if __name__ == '__main__':
    main()