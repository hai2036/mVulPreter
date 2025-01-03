import os
from re import I
from typing import Container
from preprocess import *
from complete_pdg import *
from slice_op import *
from json_to_dot import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/content/mVulPreter/dataset/')
    args = parser.parse_args()
    dataset_path = args.input_dir
    json_file = 'dataset_test_line_json/'
    ddg_dot_file = 'dataset_test_pdg_dot/'
    
    pdg_dot_path = 'dataset_test_pdg_dot_slice'
    sub_graph_path = 'subgraph_json/'
    label_pkl = 'test_label_pkl.pkl'
    label_path = dataset_path + label_pkl
    #所有数据
    container = joern_process(dataset_path+json_file)
    i = 0
    sub_cnt = 0
    for data in container:
        i += 1
        if data == []:
            sub_cnt += 1
            continue
        data = data[0]
        data_nodes = {}
        idx = data[0]
        cpg = data[1]
        print("===========>>>>>  " + str(i))
        print(idx)

        ddg_edge_list = ddg_edge_genearate(dataset_path+ddg_dot_file, idx)
        data_nodes_tmp = parse_to_nodes(cpg)
        data_nodes = complete_pdg(data_nodes_tmp, ddg_edge_list)

        generate_complete_json(data_nodes, dataset_path + pdg_dot_path, idx)
        pointer_node_list = get_pointers_node(data_nodes)
        print("pointer node list")
        print(pointer_node_list)
        if pointer_node_list != []:
            _pointer_slice_list = pointer_slice(data_nodes, pointer_node_list)
            points_name = '@pointer'
            print(_pointer_slice_list)
            generate_sub_json(data_nodes, _pointer_slice_list, dataset_path + sub_graph_path, idx, points_name, label_path)
        
        arr_node_list = get_all_array(data_nodes)
        print("arr_node_list")
        print(arr_node_list)
        if arr_node_list != []:
            _arr_slice_list = array_slice(data_nodes, arr_node_list)
            points_name = '@array'
            print(_arr_slice_list)
            generate_sub_json(data_nodes, _arr_slice_list, dataset_path + sub_graph_path, idx, points_name, label_path)

        integer_node_list = get_all_integeroverflow_point(data_nodes)
        print("integer_node_list")
        print(integer_node_list)
        if integer_node_list != []:
            _integer_slice_list = inte_slice(data_nodes, integer_node_list)
            points_name = '@integer'
            print(_integer_slice_list)
            generate_sub_json(data_nodes, _integer_slice_list, dataset_path + sub_graph_path, idx, points_name, label_path)

        call_node_list = get_all_sensitiveAPI(data_nodes)
        print("call_node_list")
        print(call_node_list)
        if call_node_list != []:
            _call_slice_list = call_slice(data_nodes, call_node_list)
            points_name = '@API'
            print(_call_slice_list)
            generate_sub_json(data_nodes, _call_slice_list, dataset_path + sub_graph_path, idx, points_name, label_path)            

if __name__ == '__main__':
    main()
