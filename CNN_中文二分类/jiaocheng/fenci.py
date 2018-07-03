# coding:utf-8
import multiprocessing
tmp_catalog = '/data/'
# file_list = [tmp_catalog+'交通214.txt', tmp_catalog+'cnews.test.txt']
file_list = [tmp_catalog+'交通214.txt']
write_list = [tmp_catalog+'交通214_cut.txt']

def tokenFile(file_path, write_path):
    word_divider = WordCut()
    with open(write_path, 'w') as w:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.decode('utf-8').strip()
                token_sen = word_divider.seg_sentence(line.split('\t')[1])
                w.write(line.split('\t')[0].encode('utf-8') + '\t' + token_sen.encode('utf-8') + '\n') 
    print file_path + ' has been token and token_file_name is ' + write_path

pool = multiprocessing.Pool(processes=4)
for file_path, write_path in zip(file_list, write_list):
    pool.apply_async(tokenFile, (file_path, write_path, ))
    print file_path
pool.close()
pool.join() # 调用join()之前必须先调用close()
print "Sub-process(es) done."