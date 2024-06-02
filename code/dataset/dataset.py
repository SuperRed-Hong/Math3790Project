# import random
# import re
# import unicodedata
#
# from torch.utils.data import Dataset
#
#
# class CWSDataset(Dataset):
#     def __init__(self, datasets: dict, train_set: bool, criterion_token: str = "[UNC]"):
#
#         self.pattern = re.compile(
#             r"\x80|…|\x08|\uecd4|\ueb78||\ue00b|\ue04b|\ue003|\ue023|\ue015|\ue06f|\ue07f|\x7f|"
#             + r"\ue3dc|\ue034|\u200b|\ue05f|\ue7fe|\u200b|\ue030|\ue771|\ue030|\ue3b2|\ue02e|\ue01f|\ue026"
#             + r"|\ue028|\ue828|\ue0b3|\ue08f|\ue008|\ue00a|\ue01d|\ue82d|\ue013|\ue017|\ue03c|\ue03a"
#         )
#
#         self.MAX_LENGTH = 509
#         self.current_state = train_set
#         self.content = []
#         self.labeled_data = []
#
#         criteria_to_number = {
#             "[AS]": 0,
#             "[CIT]": 1,
#             "[MSR]": 2,
#             "[PKU]": 3,
#             "[CTB6]": 4,
#             "[CNC]": 5,
#             "[SXU]": 6,
#             "[UD]": 7,
#             "[WTB]": 8,
#             "[ZX]": 9,
#         }
#         for dataset_name, dataset_paths in datasets.items():
#             for dataset_path in dataset_paths:
#
#                 # Read the data for training.
#                 with open(f"{dataset_path}", "r", encoding="utf8") as f:
#                     content = f.readlines()
#
#                 # Prepare the data for training
#                 if train_set:
#                     content, _ = self.split_string(content)
#                     answer = list(map(lambda x: x.split(), content))
#                     self.labeled_data += self.labelling(
#                         dataset_label=criteria_to_number[dataset_name], answer=answer
#                     )
#                     content = ["".join(self.str_normalize(data)) for data in answer]
#                     content = [[x for x in data] for data in content]
#                     self.content += [
#                         dataset_name + " " + self.transform(" ".join(data))
#                         for data in content
#                     ]
#
#                 # Prepare the data for testing
#                 else:
#                     # These are the pre-defined criteria tokens.
#                     if criterion_token not in [
#                         "[AS]",
#                         "[CIT]",
#                         "[CNC]",
#                         "[CTB6]",
#                         "[MSR]",
#                         "[PKU]",
#                         "[SXU]",
#                         "[UD]",
#                         "[WTB]",
#                         "[ZX]",
#                         "[UNC]",
#                     ]:
#                         raise ValueError(
#                             f"The criterion_token couldn't be {criterion_token}"
#                         )
#
#                     self.origin, self.short_input = self.split_string(content)
#                     self.origin = self.prepare_test_data(self.origin)
#                     self.content = [
#                         [x for x in self.str_normalize(data)] for data in self.origin
#                     ]
#                     self.content = [
#                         criterion_token + " " + self.transform(" ".join(data))
#                         for data in self.content
#                     ]
#
#     @staticmethod
#     def labelling(dataset_label, answer):
#         """
#             labelling the data
#
#             B : begin of a word
#             M : middle of a word
#             E : end of a word
#             S : a single word
#
#         Args:
#             dataset_label (_type_): _description_
#             answer (_type_): _description_
#
#         Returns:
#             _type_: _description_
#         """
#         labeled = []
#         transform = {"B": 0, "M": 1, "E": 2, "S": 3}
#         for data in answer:
#             tmp = []
#             for data_seg in data:
#                 if len(data_seg) == 1:
#                     tmp.append("S")
#                 elif len(data_seg) == 2:
#                     tmp.append("BE")
#                 else:
#                     tmp.append("B" + ("M" * (len(data_seg) - 2)) + "E")
#             labeled.append(
#                 [dataset_label] + [transform[label] for label in "".join(tmp)]
#             )
#         return labeled
#
#     @staticmethod
#     def find_period(max_length, string):
#         """_summary_
#
#         Args:
#             max_length (_type_): _description_
#             string (_type_): _description_
#
#         Returns:
#             _type_: _description_
#         """
#         punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
#         counter = 0
#         tmp = re.split(r"\s+", string)
#         for idx, data in enumerate(tmp):
#             if counter + len(data) > max_length:
#                 break
#             else:
#                 counter += len(data)
#         idx -= 1
#         max_length_idx = idx
#         while idx > 0:
#             if punctuation.match(tmp[idx]):
#                 return " ".join(tmp[: idx + 1]), " ".join(tmp[idx + 1:])
#             else:
#                 idx -= 1
#         return " ".join(tmp[: max_length_idx + 1]), " ".join(tmp[max_length_idx + 1:])
#
#     def split_string(self,dataset: list):
#         """_summary_
#             Some data in [PKU], [MSR], and [SXU] dataset is longer than 512. In       \
#             other words, it couldn't be the input of Bert, so we need the to find the \
#             nearest break point for the data and split the data into 2 pieces.
#
#         Args:
#             dataset (list): _description_
#
#         Returns:
#             _type_: _description_
#         """
#
#         space_pattern = re.compile(r"\s+")
#         retrun_data = []
#         short_input = []
#         for index in range(len(dataset)):
#             if len("".join(re.split(r"\s+", dataset[index].strip()))) > 509:
#                 data = space_pattern.sub(" ", dataset[index].strip())
#                 data, rest_of_data = self.find_period(509, data)
#                 retrun_data.append(data.strip())
#                 short_input.append(0)
#                 while (
#                         len("".join(re.split(r"\s+", rest_of_data.strip())))
#                         > 509
#                 ):
#                     data, rest_of_data = find_period(509, rest_of_data)
#                     short_input.append(0)
#                     retrun_data.append(data.strip())
#                 retrun_data.append(rest_of_data.strip())
#                 short_input.append(1)
#             else:
#                 retrun_data.append(space_pattern.sub(" ", dataset[index].strip()))
#                 short_input.append(1)
#         return retrun_data, short_input
#
#     #@staticmethod
#     # def find_period(max_length, string):
#     #     """_summary_
#     #
#     #     Args:
#     #         max_length (_type_): _description_
#     #         string (_type_): _description_
#     #
#     #     Returns:
#     #         _type_: _description_
#     #     """
#     #     punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
#     #     counter = 0
#     #     tmp = re.split(r"\s+", string)
#     #     #tmp = re.split(r"\W+", string)
#     #     # print(tmp)
#     #     for idx, data in enumerate(tmp):
#     #         if counter + len(data) > max_length:
#     #             break
#     #         else:
#     #             counter += len(data)
#     #     idx -= 1
#     #     max_length_idx = idx
#     #     # print(idx)
#     #     while idx > 0:
#     #         if punctuation.match(tmp[idx]):
#     #             return " ".join(tmp[: idx + 1]), " ".join(tmp[idx + 1 :])
#     #         else:
#     #             idx -= 1
#     #     return " ".join(tmp[: max_length_idx + 1]), " ".join(tmp[max_length_idx + 1 :])
#     #
#     # def split_string(self, dataset: list):
#     #     """_summary_
# 	# 		Some data in [PKU], [MSR], and [SXU] dataset is longer than 512. In       \
# 	# 		other words, it couldn't be the input of Bert, so we need the to find the \
# 	# 		nearest break point for the data and split the data into 2 pieces.
#     #
# 	# 	Args:
# 	# 		dataset (list): _description_
#     #
# 	# 	Returns:
# 	# 		_type_: _description_
# 	# 	"""
#     #
#     #     # print("!!!!!!!!!!!!!!!!!!!!!")
#     #     # print(len(dataset))
#     #     space_pattern = re.compile(r"\s+")
#     #     retrun_data = []
#     #     short_input = []
#     #     for index in range(len(dataset)):
#     #         # print(index)
#     #         # print(dataset[index])
#     #         if len("".join(re.split(r"\s+", dataset[index].strip()))) > self.MAX_LENGTH:
#     #             data = space_pattern.sub(" ", dataset[index].strip())
#     #             data, rest_of_data = self.find_period(self.MAX_LENGTH, data)
#     #             # print(str(1)+data)
#     #             # print(str(2)+rest_of_data)
#     #             retrun_data.append(data.strip())
#     #             short_input.append(0)
#     #             while (
#     #                 len("".join(re.split(r"\s+", rest_of_data.strip())))
#     #                 > self.MAX_LENGTH
#     #             ):
#     #                 data, rest_of_data = self.find_period(self.MAX_LENGTH, rest_of_data)
#     #                 short_input.append(0)
#     #                 retrun_data.append(data.strip())
#     #             retrun_data.append(rest_of_data.strip())
#     #             short_input.append(1)
#     #         else:
#     #             retrun_data.append(space_pattern.sub(" ", dataset[index].strip()))
#     #             short_input.append(1)
#     #     return retrun_data, short_input
#
#     def transform(self, string):
#         return self.pattern.sub("[UNK]", string)
#
#     def str_normalize(self, origin):
#         new_str = []
#         for string in origin:
#             new_str.append(unicodedata.normalize("NFD", string))
#         return new_str
#
#     def prepare_test_data(self, origin):
#         space_pattern = re.compile(r"\s+")
#         test_data = []
#         punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
#         for idx, data in enumerate(origin):
#             tmp = []
#             hold = ""
#             data = data.split()
#             for _, word in enumerate(data):
#                 if len(punctuation.sub("", word)) == 0:
#                     tmp.append(hold)
#                     tmp.append(word)
#                     hold = ""
#                 else:
#                     hold += word
#             if hold != "":
#                 tmp.append(hold)
#             test_data.append(space_pattern.sub(" ", " ".join(tmp)))
#         return test_data
#
#     def __getitem__(self, index):
#         if self.current_state:
#             return self.content[index], self.labeled_data[index]
#         else:
#             return self.content[index], self.origin[index], self.short_input[index]
#
#     def __len__(self):
#         return len(self.content)
#
#
# def find_period(max_length, string):
#     """_summary_
#
#     Args:
#         max_length (_type_): _description_
#         string (_type_): _description_
#
#     Returns:
#         _type_: _description_
#     """
#     punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
#     counter = 0
#     tmp = re.split(r"\s+", string)
#     for idx, data in enumerate(tmp):
#         if counter + len(data) > max_length:
#             break
#         else:
#             counter += len(data)
#     idx -= 1
#     max_length_idx = idx
#     while idx > 0:
#         if punctuation.match(tmp[idx]):
#             return " ".join(tmp[: idx + 1]), " ".join(tmp[idx + 1 :])
#         else:
#             idx -= 1
#     return " ".join(tmp[: max_length_idx + 1]), " ".join(tmp[max_length_idx + 1 :])
#
#
# def split_string( dataset: list):
#     """_summary_
#         Some data in [PKU], [MSR], and [SXU] dataset is longer than 512. In       \
#         other words, it couldn't be the input of Bert, so we need the to find the \
#         nearest break point for the data and split the data into 2 pieces.
#
#     Args:
#         dataset (list): _description_
#
#     Returns:
#         _type_: _description_
#     """
#
#     space_pattern = re.compile(r"\s+")
#     retrun_data = []
#     short_input = []
#     for index in range(len(dataset)):
#         if len("".join(re.split(r"\s+", dataset[index].strip()))) > 509:
#             data = space_pattern.sub(" ", dataset[index].strip())
#             data, rest_of_data = find_period(509, data)
#             retrun_data.append(data.strip())
#             short_input.append(0)
#             while (
#                 len("".join(re.split(r"\s+", rest_of_data.strip())))
#                 > 509
#             ):
#                 data, rest_of_data = find_period(509, rest_of_data)
#                 short_input.append(0)
#                 retrun_data.append(data.strip())
#             retrun_data.append(rest_of_data.strip())
#             short_input.append(1)
#         else:
#             retrun_data.append(space_pattern.sub(" ", dataset[index].strip()))
#             short_input.append(1)
#     return retrun_data, short_input
#
# if __name__ == "__main__":
#     strs=['辞别过去的百年，迈进新的世纪，我们深感责任重大。当今世界，经济全球化趋势日益增强，'
#                       '以信息技术为核心的科技革命迅猛发展，国际竞争更加激烈，形势逼人，时不我待。进入新世纪，'
#                       '我们国家将开始实施第三步战略部署，加快推进现代化，实现经济腾飞和社会全面进步。美好的前景催人奋进。'
#                       '在新的形势下，民主党派、工商联和无党派人士任重而道远。我们要在中国共产党的领导下，牢牢把握大团结、大联合的主题，'
#                       '团结和带领广大成员和所联系的群众，为建设有中国特色社会主义的经济、政治、文化服务，为维护安定团结的政治局面服务，'
#                       '为实现祖国完全繁荣昌盛的目标而努力奋斗。']
#
#     # tmp = re.split(r"\W+", strs)
#     # print(tmp)
#     # import re
#     # def split_text_with_punctuation(text):
#     #     # 使用正则表达式 \b 匹配单词边界，保留标点符号
#     #     substrings = re.split(r"(\b)", text)
#     #     # 过滤掉空字符串
#     #     substrings = [substring for substring in substrings if substring]
#     #     return substrings
#
#
#     # 示例文本
#     #input_text = "这是一段示例文本，包含一些标点符号。分割这段文本，使每个部分都以标点符号结束！"
#     print(type(strs))
#     # 按照标点符号分割文本并保留标点符号
#     result = split_string(strs)
#     # 打印结果
#     print(result)

import random
import re
import unicodedata

from torch.utils.data import Dataset


class CWSDataset(Dataset):
    def __init__(self, datasets: dict, train_set: bool, criterion_token: str = "[UNC]"):

        self.pattern = re.compile(
            r"\x80|…|\x08|\uecd4|\ueb78||\ue00b|\ue04b|\ue003|\ue023|\ue015|\ue06f|\ue07f|\x7f|"
            + r"\ue3dc|\ue034|\u200b|\ue05f|\ue7fe|\u200b|\ue030|\ue771|\ue030|\ue3b2|\ue02e|\ue01f|\ue026"
            + r"|\ue028|\ue828|\ue0b3|\ue08f|\ue008|\ue00a|\ue01d|\ue82d|\ue013|\ue017|\ue03c|\ue03a"
        )

        self.MAX_LENGTH = 509
        self.current_state = train_set
        self.content = []
        self.labeled_data = []

        criteria_to_number = {
            "[AS]": 0,
            "[CIT]": 1,
            "[MSR]": 2,
            "[PKU]": 3,
            "[CTB6]": 4,
            "[CNC]": 5,
            "[SXU]": 6,
            "[UD]": 7,
            "[WTB]": 8,
            "[ZX]": 9,
        }
        for dataset_name, dataset_paths in datasets.items():
            for dataset_path in dataset_paths:

                # Read the data for training.
                with open(f"{dataset_path}", "r", encoding="utf8") as f:
                    content = f.readlines()

                # Prepare the data for training
                if train_set:
                    content, _ = self.split_string(content)
                    answer = list(map(lambda x: x.split(), content))
                    self.labeled_data += self.labelling(
                        dataset_label=criteria_to_number[dataset_name], answer=answer
                    )
                    content = ["".join(self.str_normalize(data)) for data in answer]
                    content = [[x for x in data] for data in content]
                    self.content += [
                        dataset_name + " " + self.transform(" ".join(data))
                        for data in content
                    ]

                # Prepare the data for testing
                else:
                    # These are the pre-defined criteria tokens.
                    if criterion_token not in [
                        "[AS]",
                        "[CIT]",
                        "[CNC]",
                        "[CTB6]",
                        "[MSR]",
                        "[PKU]",
                        "[SXU]",
                        "[UD]",
                        "[WTB]",
                        "[ZX]",
                        "[UNC]",
                    ]:
                        raise ValueError(
                            f"The criterion_token couldn't be {criterion_token}"
                        )

                    self.origin, self.short_input = self.split_string(content)
                    self.origin = self.prepare_test_data(self.origin)
                    self.content = [
                        [x for x in self.str_normalize(data)] for data in self.origin
                    ]
                    self.content = [
                        criterion_token + " " + self.transform(" ".join(data))
                        for data in self.content
                    ]

    @staticmethod
    def labelling(dataset_label, answer):
        """
            labelling the data

            B : begin of a word
            M : middle of a word
            E : end of a word
            S : a single word

        Args:
            dataset_label (_type_): _description_
            answer (_type_): _description_

        Returns:
            _type_: _description_
        """
        labeled = []
        transform = {"B": 0, "M": 1, "E": 2, "S": 3}
        for data in answer:
            tmp = []
            for data_seg in data:
                if len(data_seg) == 1:
                    tmp.append("S")
                elif len(data_seg) == 2:
                    tmp.append("BE")
                else:
                    tmp.append("B" + ("M" * (len(data_seg) - 2)) + "E")
            labeled.append(
                [dataset_label] + [transform[label] for label in "".join(tmp)]
            )
        return labeled

    @staticmethod
    def find_period(max_length, string):
        """_summary_

        Args:
            max_length (_type_): _description_
            string (_type_): _description_

        Returns:
            _type_: _description_
        """
        punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
        counter = 0
        tmp = re.split(r"\s+", string)
        for idx, data in enumerate(tmp):
            if counter + len(data) > max_length:
                break
            else:
                counter += len(data)
        idx -= 1
        max_length_idx = idx
        while idx > 0:
            if punctuation.match(tmp[idx]):
                return " ".join(tmp[: idx + 1]), " ".join(tmp[idx + 1 :])
            else:
                idx -= 1
        return " ".join(tmp[: max_length_idx + 1]), " ".join(tmp[max_length_idx + 1 :])

    def split_string(self, dataset: list):
        """_summary_
			Some data in [PKU], [MSR], and [SXU] dataset is longer than 512. In       \
			other words, it couldn't be the input of Bert, so we need the to find the \
			nearest break point for the data and split the data into 2 pieces.

		Args:
			dataset (list): _description_

		Returns:
			_type_: _description_
		"""

        space_pattern = re.compile(r"\s+")
        retrun_data = []
        short_input = []
        for index in range(len(dataset)):
            if len("".join(re.split(r"\s+", dataset[index].strip()))) > self.MAX_LENGTH:
                data = space_pattern.sub(" ", dataset[index].strip())
                data, rest_of_data = self.find_period(self.MAX_LENGTH, data)
                retrun_data.append(data.strip())
                short_input.append(0)
                while (
                    len("".join(re.split(r"\s+", rest_of_data.strip())))
                    > self.MAX_LENGTH
                ):
                    data, rest_of_data = self.find_period(self.MAX_LENGTH, rest_of_data)
                    short_input.append(0)
                    retrun_data.append(data.strip())
                retrun_data.append(rest_of_data.strip())
                short_input.append(1)
            else:
                retrun_data.append(space_pattern.sub(" ", dataset[index].strip()))
                short_input.append(1)
        return retrun_data, short_input

    def transform(self, string):
        return self.pattern.sub("[UNK]", string)

    def str_normalize(self, origin):
        new_str = []
        for string in origin:
            new_str.append(unicodedata.normalize("NFD", string))
        return new_str

    def prepare_test_data(self, origin):
        space_pattern = re.compile(r"\s+")
        test_data = []
        punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
        for idx, data in enumerate(origin):
            tmp = []
            hold = ""
            data = data.split()
            for _, word in enumerate(data):
                if len(punctuation.sub("", word)) == 0:
                    tmp.append(hold)
                    tmp.append(word)
                    hold = ""
                else:
                    hold += word
            if hold != "":
                tmp.append(hold)
            test_data.append(space_pattern.sub(" ", " ".join(tmp)))
        return test_data

    def __getitem__(self, index):
        if self.current_state:
            return self.content[index], self.labeled_data[index]
        else:
            return self.content[index], self.origin[index], self.short_input[index]

    def __len__(self):
        return len(self.content)
