import pandas as pd
import random
import copy
import numpy as np
from scipy.optimize import linprog
from retrying import retry
import warnings

# warnings.filterwarnings("ignore")
data_path = "./filter_data.csv"


class TupleInfo:
    def __init__(self):
        self.attri_values = dict()


class MutualCover:
    def __init__(self, original_data):
        self.original_data = original_data
        self.output_path = "./results/mutual_cover/diversity/mutual_"
        self.l_values = [10, 12, 15, 18, 20]
        # self.k_values = [5, 8, 10]
        self.k_values = [6, 7]
        self.seed_nums = range(10)

    def oper_start(self):
        self.read_attri_names()
        self.read_tuple_info()
        self.count_qi_range()
        for lv in self.l_values:
            print("l_value=" + str(lv))
            self.lv = lv
            self.group_id = 1
            self.anonymized_data = dict()
            for kv in self.k_values:
                self.anonymized_data[kv] = dict()
                for sd in self.seed_nums:
                    self.anonymized_data[kv][sd] = pd.DataFrame(columns=self.attri_names + ["GID", "OID", "PEN"])
            self.table_arr = [range(0, len(self.tuple_info)), ]
            self.ori_number = len(self.tuple_info)
            while len(self.table_arr) > 0:
                self.oper_tuples = self.table_arr.pop()
                self.mon_start()
            self.save_result()
        return

    def read_attri_names(self):
        self.attri_names = list(self.original_data.columns)
        self.qi_attributes = ["RELATE", "SEX", "AGE", "MARST", "RACE", "EDUC", "UHRSWORK"]
        return

    def read_tuple_info(self):
        self.tuple_info = {}
        for row_index, row_info in self.original_data.iterrows():
            info = TupleInfo()
            for colu_name, colu_value in row_info.items():
                info.attri_values[colu_name] = colu_value
            self.tuple_info[row_index] = info
        return

    def count_qi_range(self):
        self.qi_range = {}
        for qattri in self.qi_attributes:
            max_value = self.original_data[qattri].max()
            min_value = self.original_data[qattri].min()
            self.qi_range[qattri] = max_value - min_value + 1
            print(qattri + ": " + str(self.qi_range[qattri]))
        return

    def mon_start(self):
        attri_spread = list()
        attri_domain = dict()
        for aname in self.qi_attributes:
            domain = self.get_attri_info(aname)
            spread = self.cal_spread(domain)
            attri_spread.append([aname, spread])
            attri_domain[aname] = domain
        attri_spread.sort(key=lambda x: x[1])
        partition_flag = False
        while partition_flag == False and len(attri_spread) != 0:
            name_spread = attri_spread.pop()
            middle_value = self.get_middle(attri_domain[name_spread[0]])
            if self.check_partition(name_spread[0], middle_value) is True:
                partition_flag = True
        if partition_flag == True:
            self.table_arr.append(self.less_id)
            self.table_arr.append(self.more_id)
        else:
            self.anatomy()
        return

    def get_attri_info(self, aname):
        domain_values = dict()
        for id in self.oper_tuples:
            if self.tuple_info[id].attri_values[aname] in domain_values:
                domain_values[self.tuple_info[id].attri_values[aname]] += 1
            else:
                domain_values[self.tuple_info[id].attri_values[aname]] = 1.0
        domain = []
        for key, value in domain_values.items():
            domain.append([key, value])
        domain.sort(key=lambda x: x[0])
        return domain

    def cal_spread(self, domain):
        sum = 0.0
        num = 0.0
        for attri in domain:
            sum += attri[0] * attri[1]
            num += attri[1]
        aver = sum / num
        squar = 0.0
        for attri in domain:
            squar += ((attri[0] - aver) ** 2) * attri[1]
        return squar

    def get_middle(self, domain):
        sum = 0
        for value in domain:
            sum += value[1]
        middle = sum / 2.0
        sum = 0
        for value in domain:
            sum += value[1]
            if sum >= middle:
                mvalue = value[0]
                break
        return mvalue

    def check_partition(self, attri_name, attri_value):
        less_set = {}
        self.less_id = []
        more_set = {}
        self.more_id = []
        for id in self.oper_tuples:
            if self.tuple_info[id].attri_values[attri_name] <= attri_value:
                if self.tuple_info[id].attri_values["INCWAGE"] in less_set:
                    less_set[self.tuple_info[id].attri_values["INCWAGE"]] += 1
                else:
                    less_set[self.tuple_info[id].attri_values["INCWAGE"]] = 1
                self.less_id.append(id)
            else:
                if self.tuple_info[id].attri_values["INCWAGE"] in more_set:
                    more_set[self.tuple_info[id].attri_values["INCWAGE"]] += 1
                else:
                    more_set[self.tuple_info[id].attri_values["INCWAGE"]] = 1
                self.more_id.append(id)
        if len(self.less_id) == 0 or len(self.more_id) == 0:
            return False
        if self.get_mvalue(less_set) * self.lv <= len(self.less_id) and self.get_mvalue(more_set) * self.lv <= len(
                self.more_id):
            return True
        else:
            return False

    def get_mvalue(self, set):
        max_value = -1
        for key in set:
            if set[key] > max_value:
                max_value = set[key]
        return max_value

    def anatomy(self):
        self.read_inc_ids()
        self.read_inc_count()
        self.partition_number = len(self.oper_tuples)
        while self.partition_number > 0:
            sen_set, set_num = self.pick_sen_set()
            while set_num > 0:
                pick_ids = list()
                for i in range(len(sen_set)):
                    pick_ids.append(self.inc_ids[sen_set[i]].pop())
                self.insert_data(pick_ids)
                set_num -= 1
        return

    def read_inc_ids(self):
        self.inc_ids = dict()
        for id in self.oper_tuples:
            if self.tuple_info[id].attri_values["INCWAGE"] in self.inc_ids:
                self.inc_ids[self.tuple_info[id].attri_values["INCWAGE"]].append(id)
            else:
                self.inc_ids[self.tuple_info[id].attri_values["INCWAGE"]] = [id, ]
        return

    def read_inc_count(self):
        self.inc_count = list()
        for incwage_value in self.inc_ids:
            self.inc_count.append([incwage_value, len(self.inc_ids[incwage_value])])
        self.inc_count.sort(key=lambda x: x[1], reverse=True)
        return

    def pick_sen_set(self):
        beta = self.lv
        alpha = self.cal_alpha(beta)
        while alpha < 0:
            beta += 1
            alpha = self.cal_alpha(beta)
        self.partition_number -= beta * alpha
        type_list = []
        for i in range(beta):
            type_name = self.inc_count[i][0]
            type_list.append(type_name)
            self.inc_count[i][1] -= alpha
        if beta != len(self.inc_count):
            self.sort_pair(beta)
        return type_list, alpha

    def cal_alpha(self, beta):
        if beta == len(self.inc_count):
            return 1
        alpha = self.inc_count[beta - 1][1]
        while alpha > 0:
            left_value = self.partition_number - alpha * beta
            con_1 = self.lv * (self.inc_count[0][1] - alpha) <= left_value
            con_2 = self.lv * self.inc_count[beta][1] <= left_value
            if con_1 and con_2:
                return alpha
            else:
                alpha -= 1
        return -1

    def sort_pair(self, beta):
        temp_arr = []
        i = 0
        j = beta
        while i < beta:
            if self.inc_count[i][1] > self.inc_count[j][1]:
                temp_arr.append(self.inc_count[i])
                i += 1
            else:
                temp_arr.append(self.inc_count[j])
                j += 1
                if j == len(self.inc_count):
                    break
        while i < beta:
            temp_arr.append(self.inc_count[i])
            i += 1
        while j < len(self.inc_count):
            temp_arr.append(self.inc_count[j])
            j += 1
        self.inc_count = temp_arr
        return

    def insert_data(self, pick_ids):
        ori_values = self.store_ori_values(pick_ids)
        group_range = self.count_group_range(ori_values)
        group_weights = self.count_group_weight(group_range)
        for kv in self.k_values:
            attri_probabilities = self.count_probability(ori_values, kv)
            for sd in self.seed_nums:
                random.seed(sd)
                insert_values = dict()
                for id in pick_ids:
                    insert_values[id] = dict()
                unchange_flag = True
                max_permute = 1
                num_permute = 0
                pen_values = dict()
                zero_pen = list()
                while unchange_flag is True and num_permute < max_permute:
                    for id in pick_ids:
                        pen_values[id] = 0
                    zero_pen = list()
                    unchange_flag = False
                    num_permute += 1
                    for attri in self.qi_attributes:
                        noise_values = self.add_noise(ori_values[attri], attri_probabilities[attri])
                        for id_index in range(len(pick_ids)):
                            insert_values[pick_ids[id_index]][attri] = noise_values[id_index]
                            pen_values[pick_ids[id_index]] += abs(
                                noise_values[id_index] - self.tuple_info[pick_ids[id_index]].attri_values[attri]) / \
                                                              self.qi_range[attri]
                    for id_index in range(len(pick_ids)):
                        if pen_values[pick_ids[id_index]] == 0:
                            unchange_flag = True
                            zero_pen.append(pick_ids[id_index])
                if unchange_flag is True:
                    self.permute_zero_tuples(zero_pen, insert_values, group_weights, group_range, pen_values)
                for i in range(len(pick_ids)):
                    temp_data = pd.DataFrame({"RELATE": insert_values[pick_ids[i]]["RELATE"],
                                              "SEX": insert_values[pick_ids[i]]["SEX"],
                                              "AGE": insert_values[pick_ids[i]]["AGE"],
                                              "MARST": insert_values[pick_ids[i]]["MARST"],
                                              "RACE": insert_values[pick_ids[i]]["RACE"],
                                              "EDUC": insert_values[pick_ids[i]]["EDUC"],
                                              "OCC": self.tuple_info[pick_ids[i]].attri_values["OCC"],
                                              "UHRSWORK": insert_values[pick_ids[i]]["UHRSWORK"],
                                              "INCWAGE": self.tuple_info[pick_ids[i]].attri_values["INCWAGE"],
                                              "OID": pick_ids[i],
                                              "PEN": pen_values[pick_ids[i]],
                                              "GID": self.group_id},
                                             index=[0])
                    self.anonymized_data[kv][sd] = self.anonymized_data[kv][sd].append(temp_data, ignore_index=True)
        self.group_id += 1
        self.ori_number -= len(pick_ids)
        print(self.ori_number)
        return

    def store_ori_values(self, ids):
        ori_values = dict()
        for attri in self.qi_attributes:
            ori_values[attri] = list()
            for id_index in range(len(ids)):
                ori_values[attri].append(self.tuple_info[ids[id_index]].attri_values[attri])
        return ori_values

    def count_group_range(self, ori_values):
        group_range = dict()
        for attri in self.qi_attributes:
            group_range[attri] = self.return_range(ori_values[attri])
        return group_range

    def return_range(self, input_values):
        min_value = input_values[0]
        max_value = input_values[0]
        for value in input_values:
            if value < min_value:
                min_value = value
            if value > max_value:
                max_value = value
        return np.array(range(min_value, max_value + 1))

    def count_group_weight(self, group_range):
        group_weights = dict()
        weight_sum = 0
        for attri in self.qi_attributes:
            group_weights[attri] = (group_range[attri].max() - group_range[attri].min()) / self.qi_range[attri]
            weight_sum += group_weights[attri]
        for attri in self.qi_attributes:
            group_weights[attri] /= weight_sum
        return group_weights

    def count_probability(self, ori_values, kv):
        attri_probabilities = dict()
        for attri in self.qi_attributes:
            # floor_number = [1]
            attri_probabilities[attri] = self.calculate_probability(ori_values[attri], kv)
        return attri_probabilities

    @retry
    def calculate_probability(self, input_values, k_value):
        column_values = self.return_range(input_values)
        c_array = list()
        for value in input_values:
            # c_array.extend(np.power(np.abs(column_values - value) + 1, 2))
            c_array.extend(np.abs(column_values - value))
        c_array = np.array(c_array, dtype='float16')

        A_eq = list()
        for i in range(len(input_values)):
            temp_array = [0] * len(input_values) * len(column_values)
            temp_array[i * len(column_values):(i + 1) * len(column_values)] = [1] * len(column_values)
            A_eq.append(temp_array)
        A_eq = np.array(A_eq, dtype='float16')
        B_eq = np.array([1] * len(input_values), dtype='float16')

        A_ub = list()
        for i in range(len(column_values)):
            temp_array = [0] * len(input_values) * len(column_values)
            for j in range(len(input_values)):
                temp_array[j * len(column_values) + i] = -1
            for j in range(len(input_values)):
                copy_array = copy.deepcopy(temp_array)
                copy_array[j * len(column_values) + i] = k_value - 1
                A_ub.append(copy_array)
        A_ub = np.array(A_ub, dtype='float16')
        B_ub = np.array([0] * len(input_values) * len(column_values), dtype='float16')

        bounds = list()
        for i in range(len(c_array)):
            bounds.append((0, None))
        r = linprog(c_array, A_ub, B_ub, A_eq, B_eq, bounds=bounds)
        # while True:
        #     bounds = [(pow(0.1, floor_number[0]), None), ] * len(c_array)
        #     floor_number[0] += 1
        #     r = linprog(c_array, A_ub, B_ub, A_eq, B_eq, bounds=bounds)
        #     sum_test = 0.0
        #     for va_test in range(len(column_values)):
        #         sum_test += r.x[va_test]
        #     if sum_test > 1:
        #         continue
        #     else:
        #         break
        probabilities = copy.deepcopy(r.x)
        probabilities = probabilities.reshape((len(input_values), len(column_values)))
        return probabilities

    def add_noise(self, input_values, probabilities):
        column_values = self.return_range(input_values)
        noise_values = list()
        for i in range(len(input_values)):
            random_value = random.random()
            sum = 0
            for j in range(len(column_values)):
                sum += probabilities[i][j]
                if random_value <= sum:
                    noise_values.append(column_values[j])
                    break
                if j == len(column_values) - 1:
                    noise_values.append(column_values[j])
        return noise_values

    def permute_zero_tuples(self, zero_pen, insert_values, group_weights, group_range, pen_values):
        attri_index = list(group_weights.keys())
        weight_values = [group_weights[attri] for attri in attri_index]
        for zero_id in zero_pen:
            while True:
                permute_index = np.random.choice(a=range(len(attri_index)), size=1, replace=False, p=weight_values)
                permute_attri = attri_index[permute_index[0]]
                permute_value = np.random.choice(group_range[permute_attri])
                if permute_value == insert_values[zero_id][permute_attri]:
                    continue
                else:
                    insert_values[zero_id][permute_attri] = permute_value
                    pen_values[zero_id] += abs(permute_value - self.tuple_info[zero_id].attri_values[permute_attri]) / self.qi_range[permute_attri]
                    break
        return

    def save_result(self):
        for kv in self.k_values:
            for sd in self.seed_nums:
                self.output_tname = self.output_path + "l" + str(self.lv) + "_k" + str(kv) + "_r" + str(sd)
                self.anonymized_data[kv][sd].to_csv(self.output_tname, sep=",", header=True, index=False)
        return


if __name__ == "__main__":
    original_data = pd.read_csv(data_path)
    mc = MutualCover(original_data)
    mc.oper_start()
