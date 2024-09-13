# adapted from https://github.com/xinjli/phonepiece/blob/master/phonepiece/timit.py


class TIMITConverter:
    def __init__(self, timit_mapping_path='data'):
        timit_path = f'{timit_mapping_path}/timit.txt'

        self.timit_map = {}

        with open(timit_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split()

                if fields[1] != 'nan':
                    self.timit_map[fields[0]] = fields[2]
                else:
                    self.timit_map[fields[0]] = ''

    def convert(self, str_or_lst):

        if isinstance(str_or_lst, str):
            str_or_lst = str_or_lst.split(' ')

        result = []
        for phone in str_or_lst:
            phone = phone.lower()

            if phone not in self.timit_map:
                print(f"{phone} is not a TIMIT phone")

            timit_phone = self.timit_map[phone]

            # skip q and silence
            if timit_phone == '' or timit_phone == 'sil':
                continue

            arpa_phone = timit_phone

            result.append(arpa_phone)

        return result
