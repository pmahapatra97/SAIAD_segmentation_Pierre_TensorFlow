
BASE_CASE_PATIENT_NAMES = ['SAIAD_01','SAIAD_02','SAIAD_02Bis','SAIAD_04','SAIAD_05','SAIAD_07','SAIAD_09',
                           'SAIAD_10','SAIAD_11','SAIAD_12','SAIAD_13','SAIAD_14','SAIAD_15','SAIAD_15Bis']


SCANNER_DIR = '../img/{}/scan/jpg/'
SCANNER_PREPROC_DIR = '../img/{}/scan/preproc/'

SEG_DIR = '../img/{}/{}/train/seg/classes/'
SEG_DIR_OHE = '../img/{}/{}/train/seg/classes_ohe/'
SEG_DIR_TEST = '../img/{}/{}/test/seg/classes/'
RESULTS_DIR_EXP = '../results/results_{}/'
RESULTS_DIR = RESULTS_DIR_EXP+'{}/'
CSV_FILE = '../res_csv/res_{}.csv'
CSV_FILE_SLICES = '../res_csv/res_{}_slices.csv'