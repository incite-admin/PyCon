import os
import numpy as np
from utils.pyct_attack_exp import get_save_dir_from_save_exp

##### Generate Inputs #####

def pyct_shap_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                }

                inputs.append(one_input)
                
    return inputs


def pyct_random_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000
    
    mnist_dataset = MnistDataset()        
    random_pixels = mnist_test_data_10000()
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"random_{ton_n}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                }

                inputs.append(one_input)
                
    return inputs

def pyct_rnn_random_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import RNN_MnistDataset
    from utils.gen_random_pixel_location import rnn_mnist_test_data_10000
    
    mnist_dataset = RNN_MnistDataset()        
    random_pixels = rnn_mnist_test_data_10000()
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"random_{ton_n}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                }

                inputs.append(one_input)
                
    return inputs

def pyct_rnn_shap_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import RNN_MnistDataset
    mnist_dataset = RNN_MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    # 'only_first_forward': True
                }

                inputs.append(one_input)
                
    return inputs

def stock_shap_1_2_3_4_8_limit_range02(model_name, first_n_img):
    from utils.dataset import IMDB_Dataset
    
    imdb_dataset = IMDB_Dataset()
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/imdb_sort_shap_pixel.npy')

    limit_p = 0.2
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/stock_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"stock_test_{idx}",
                    "exp_name": f"limit_{limit_p}/shap_{ton_n_shap}",
                    "save_smt": True
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = stock_dataset.get_stock_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                
    return inputs


def stock_random_1_2_3_4_8_range02(model_name, first_n_img):
    from utils.dataset import MSstock_Dataset
    from utils.gen_random_pixel_location import lstm_stock_strategy_502
    
    stock_dataset = MSstock_Dataset()        
    random_pixels = lstm_stock_strategy_502()
    limit_p = 0.2
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"stock_test_{idx}",
                    "exp_name": f"limit_{limit_p}/random_{ton_n}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = stock_dataset.get_stock_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                
    return inputs

def imdb_shap_1_2_3_4_8_range02(model_name, first_n_img):
    from utils.dataset import IMDB_Dataset
    from utils.gen_random_pixel_location import lstm_imdb_30
    
    imdb_dataset = IMDB_Dataset()
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/imdb_sort_shap_pixel.npy')
    limit_p = 0.2
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"imdb_test_{idx}",
                    "exp_name": f"limit_{limit_p}/shap_{ton_n}",
                    # "save_smt": True
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n].tolist()
                in_dict, con_dict = imdb_dataset.get_imdb_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                
    return inputs

def imdb_shap_1_2_3_4_8_range02(model_name, first_n_img):
    from utils.dataset import IMDB_Dataset
    from utils.gen_random_pixel_location import lstm_imdb_30
    
    random_pixels = lstm_imdb_30()
    limit_p = 0.2
    
    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"imdb_test_{idx}",
                    "exp_name": f"limit_{limit_p}/random_{ton_n}",
                    # "save_smt": True
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = imdb_dataset.get_imdb_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                
    return inputs


def cnn678_stack_shap_8_compare2deepconcolic_1000input(model_name):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    ### cnn678 與 deepconcolic 的 1000 張圖比較
    cnn678_index = np.load('./utils/dataset/cnn678_1000_index.npy')
    
    inputs = []

    for solve_order_stack in [True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [8]:
            
            for idx in cnn678_index:
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"compare2deepconcolic_shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                }

                inputs.append(one_input)
                
    return inputs


def cnn2418_stack_shap_8_compare2deepconcolic_1000input(model_name):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    ### cnn2418 與 deepconcolic 的 1000 張圖比較
    cnn2418_index = np.load('./utils/dataset/cnn2418_1000_index.npy')
    
    inputs = []

    for solve_order_stack in [True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [8]:
            
            for idx in cnn2418_index:
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"compare2deepconcolic_shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                }

                inputs.append(one_input)
                
    return inputs
