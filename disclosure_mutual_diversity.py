import pandas as pd
import random


random.seed(42)


class CheckDisclosure:
    def __init__(self):
        self.ori_path = "./filter_data.csv"                                                         # 原始数据路径
        self.mutual_path = "./results/mutual_cover/diversity/mutual_"                               # 匿名数据路径
        self.output_path = "./results/mutual_cover/diversity/disclosure"                            # 结果输出路径
        self.seed_numbers = range(0, 10)                        # 随机种子
        self.l_values = [10, 12, 15, 18, 20]                    # l参数
        self.k_values = [6, 7]                              # k参数
        self.probabilities = [0.3, 0.5, 0.8, 1]                 # 攻击者保留QI值的概率
        self.qi_attributes = ["RELATE", "SEX", "AGE", "MARST", "RACE", "EDUC", "UHRSWORK"]          # 作为匹配条件的属性

    def check_oper(self):
        self.original_data = pd.read_csv(self.ori_path)  # 读取原始数据
        for pr in self.probabilities:
            self.generate_conditions(pr)
            print("pr_value=:" + str(pr))
            for kv in self.k_values:
                self.kv = kv
                print("k_value=" + str(kv))
                for lv in self.l_values:
                    self.lv = lv
                    print("l_value=" + str(lv))
                    mutual_data = self.read_mutual_data()
                    identity_disclosure = [0.0, ] * len(self.seed_numbers)  # 每个随机种子下的身份暴露概率
                    attribute_disclosure = [0.0, ] * len(self.seed_numbers)  # 每个随机种子下的敏感值暴露概率
                    for id_index, row_info in self.original_data.iterrows():
                        if id_index % 100 == 0:
                            print(id_index)
                        ori_svalue = row_info["INCWAGE"]  # 记录原始数据的敏感值
                        # ori_match = self.return_match_data(original_data, temp_condition)
                        # ori_match_number = ori_match.shape[0]
                        temp_condition = self.conditions[id_index]
                        for md_index in range(len(mutual_data)):
                            md = mutual_data[md_index]  # 取出相应的匿名数据
                            md_match, contain_flag = self.return_match_data(md, temp_condition,
                                                                            id_index)  # 根据匹配条件筛选出满足条件的tuple
                            md_match_number = md_match.shape[0]
                            if md_match_number == 0:  # 如果没有满足条件的tuple，则继续下一轮
                                continue
                            if contain_flag == True:
                                identity_disclosure[md_index] += 1 / md_match_number
                            attribute_disclosure[md_index] += self.count_attri_disclosure(ori_svalue, md_match)
                    identity_disclosure = [i / self.original_data.shape[0] for i in identity_disclosure]
                    attribute_disclosure = [i / self.original_data.shape[0] for i in attribute_disclosure]
                    ide_dis_max, ide_dis_min, ide_dis_avg = self.count_max_min_avg(identity_disclosure)
                    att_dis_max, att_dis_min, att_dis_avg = self.count_max_min_avg(attribute_disclosure)
                    self.store_results(pr, kv, lv, ide_dis_max, ide_dis_min, ide_dis_avg, att_dis_max, att_dis_min,
                                       att_dis_avg)
        return

    def generate_conditions(self, pr):
        self.conditions = dict()
        for id_index, row_info in self.original_data.iterrows():
            self.conditions[id_index] = dict()
            while len(self.conditions[id_index]) == 0:
                for qi_col in self.qi_attributes:
                    if random.random() <= pr:
                        self.conditions[id_index][qi_col] = row_info[qi_col]
        return

    def read_mutual_data(self):
        mutual_data = []
        for sd in self.seed_numbers:
            mutual_tname = self.mutual_path + "l" + str(self.lv) + "_k" + str(self.kv) + "_r" + str(sd)
            mdata = pd.read_csv(mutual_tname)
            mutual_data.append(mdata)
        return mutual_data

    def return_match_data(self, ori_data, condition_set, id_index):
        temp_flags = None
        for col in condition_set:
            if temp_flags is None:
                temp_flags = (ori_data[col] == condition_set[col])
                continue
            temp_flags &= (ori_data[col] == condition_set[col])
        match_data = ori_data.loc[temp_flags, self.qi_attributes + ["INCWAGE", "OID"]].copy()
        contain_flag = True
        if match_data[match_data["OID"] == id_index].shape[0] == 0:
            contain_flag = False
        return match_data, contain_flag

    def count_attri_disclosure(self, ori_svalue, md_match):
        temp_count = md_match["INCWAGE"].value_counts()
        match_count = 0
        for col, val in temp_count.items():
            if ori_svalue == col:
                match_count += val
                break
        return match_count / md_match.shape[0]

    def count_max_min_avg(self, dis_values):
        max_value = dis_values[0]
        min_value = dis_values[0]
        avg_value = 0
        for val in dis_values:
            if val > max_value:
                max_value = val
            if val < min_value:
                min_value = val
            avg_value += val
        return max_value, min_value, avg_value / len(dis_values)

    def store_results(self, pr, kv, lv, ide_dis_max, ide_dis_min, ide_dis_avg, att_dis_max, att_dis_min, att_dis_avg):
        with open(self.output_path, "a") as tfile:
            tfile.write("Identity Disclosure: k=" + str(kv) + " l=" + str(lv) + " p=" + str(pr) + ":\n")
            tfile.write("max: " + str(ide_dis_max) + "\n")
            tfile.write("min: " + str(ide_dis_min) + "\n")
            tfile.write("avg: " + str(ide_dis_avg) + "\n")
            tfile.write("Attribute Disclosure: k=" + str(kv) + " l=" + str(lv) + " p=" + str(pr) + ":\n")
            tfile.write("max: " + str(att_dis_max) + "\n")
            tfile.write("min: " + str(att_dis_min) + "\n")
            tfile.write("avg: " + str(att_dis_avg) + "\n")
            tfile.write("--------------------------------------------------------------------\n")
        return


if __name__ == "__main__":
    cd = CheckDisclosure()
    cd.check_oper()
