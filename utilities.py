import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

"""
This class contains a list of useful helper functions which can assist the major classes with
tasks like array transformations, graph display, etc.
"""
class Utilities():

    """
    Given start and end points, returns a numpy array with all points 
    in between (including start and end points)
    """
    @staticmethod
    def fill_array(start_point, end_point):
        output_array = []
        reverse = False
        # Generate Output Array, when Start and end points are same
        if ((start_point[0] == end_point[0]) and (start_point[1] == end_point[1])):
            output_array.append(start_point)
        # Generate Output Array, when Points lie on Vertical axis
        elif (start_point[0] == end_point[0]):
            # Swap points if start point lies after end point
            if start_point[1] > end_point[1]:
                reverse = True
                temp_point = start_point
                start_point = end_point
                end_point = temp_point
            # Iterate and Generate Intermediate Points
            for i in range(start_point[1], end_point[1]+1):
                output_array.append([start_point[0], i])
        # Generate Output Array, when Points lie on Horizontal axis
        elif (start_point[1] == end_point[1]):
            # Swap points if start point lies after end point
            if start_point[0] > end_point[0]:
                reverse = True
                temp_point = start_point
                start_point = end_point
                end_point = temp_point
            # Iterate and Generate Intermediate Points
            for i in range(start_point[0], end_point[0]+1):
                output_array.append([i, start_point[1]])
        # Reverse Array when required
        if reverse:
            output_array = output_array[::-1]
        # Convert array to numpy format
        output_array = np.array(output_array)
        # Output the generated array
        return output_array
    
    """
    Merge arrays to return a unique numpy array containing all elements
    """
    @staticmethod
    def merge_arrays(array_list):
        output_array = []
        for curr_array in array_list:
            output_array.extend(curr_array.tolist())
        output_array = Utilities.remove_duplicate_points(output_array)
        output_array = np.array(output_array)
        return output_array
    
    """
    Remove duplicate entries from given list
    """
    def remove_duplicate_points(array_list):
        unique_list = []
        for sub_array in array_list:
            if sub_array not in unique_list:
                unique_list.append(sub_array)
        return unique_list
    
    """
    Orders the elements of a numpy array in ascending order and returns the ordered numpy array.
    If inverse True, return numpy array containing the remaining elements 
    from the complete array of given shape.
    """
    def order_array(shape, given_array, inverse=False):
        ordered_array = []
        rest_array = []
        x_size = shape[0]
        y_size = shape[1]
        input_list = given_array.tolist()
        for x in np.arange(x_size):
            for y in np.arange(y_size):
                curr_item = [x,y]
                if curr_item in input_list:
                    ordered_array.append(curr_item)
                else:
                    rest_array.append(curr_item)
        if inverse:
            return np.array(rest_array)
        return np.array(ordered_array)
    
    """
    Returns Symbol matrix (where symbols define observation states) in String form
    """
    def get_symbol_matrix_string(given_array, symbols=None):
        output_str = f''
        x_size = given_array.shape[0]
        y_size = given_array.shape[1]
        input_array = given_array.tolist()
        for y in np.array(range(y_size-1,-1,-1)):
            output_str += f'|'
            for x in np.arange(x_size):
                curr_symbol = symbols[int(input_array[x][y])] if symbols is not None else int(input_array[x][y])
                output_str += f' {curr_symbol} |'
            output_str += f'\n'
        return output_str
    
    """
    Get reward matrix for given array
    """
    def get_reward_matrix(given_array, find_val=2, replace_found=5.0, replace_not_found=0.0):
        x_size = given_array.shape[0]
        y_size = given_array.shape[1]
        reward_matrix = [[float(replace_found) if (given_array[x][y]==find_val) else float(replace_not_found)
                            for y in np.arange(y_size)] 
                        for x in np.arange(x_size)]
        reward_matrix = np.array(reward_matrix)
        return reward_matrix

    """
    Checks if array in list of arrays, provided needle and all items in haystack are of the same dimension
    """
    def item_in_array(haystack, needle):
        if (haystack is None) or (haystack == []) or (needle is None) or (needle == []):
            return False
        for item in haystack:
            if (len(item) != len(needle)):
                return False
            is_item_in_array = True
            for i in range(len(item)):
                if (int(item[i]) != int(needle[i])):
                    is_item_in_array = False
                    break
            if is_item_in_array:
                return True
        return False
    
    """
    Displays a 2D Plot with given data and label parameters
    """
    def display_graph(x_data, y_data, x_label='X-Axis', y_label='Y-Axis', title='Custom Graph', show_average=False):
        style.use("ggplot")

        plt.plot(x_data, y_data, label="Individual values")
        plt.title(title, fontsize=18)

        if show_average:
            avg_val = sum(y_data)/len(y_data)
            x_avg_data = x_data
            y_avg_data = [avg_val for i in range(len(x_avg_data))]
            plt.plot(x_avg_data, y_avg_data, label="Average value")
            plt.legend()
        
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)

        plt.show()
    