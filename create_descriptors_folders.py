

import os

list_folders= ['ntu120_ADD_test_csetup' ,  'ntu120_FDD_train_csub' ,  'ntu120_JDD_train_csetup' , 'ntu60_ADD_test_cview'  , 'ntu60_GD_test_csub'  ,  'ntu60_JDD_train_cview',  'ucla_FDD_test'  , 'ucla_LAD_train',
'ntu120_ADD_test_csub'   ,  'ntu120_GD_test_csetup' ,  'ntu120_JDD_train_csub'   , 'ntu60_ADD_train_csub' ,  'ntu60_GD_test_cview' ,  'ntu60_LAD_test_csub'   , 'ucla_FDD_train',
'ntu120_ADD_train_csetup' , 'ntu120_GD_test_csub' ,    'ntu120_LAD_test_csetup' ,  'ntu60_ADD_train_cview' , 'ntu60_GD_train_csub'  , 'ntu60_LAD_test_cview' ,  'ucla_GD_test',
'ntu120_ADD_train_csub'  ,  'ntu120_GD_train_csetup',  'ntu120_LAD_test_csub'  ,   'ntu60_FDD_test_csub'  ,  'ntu60_GD_train_cview',  'ntu60_LAD_train_csub' ,  'ucla_GD_train',
'ntu120_FDD_test_csetup'  , 'ntu120_GD_train_csub'  ,  'ntu120_LAD_train_csetup',  'ntu60_FDD_test_cview' ,  'ntu60_JDD_test_csub'  , 'ntu60_LAD_train_cview' , 'ucla_JDD_test',
'ntu120_FDD_test_csub'   ,  'ntu120_JDD_test_csetup' , 'ntu120_LAD_train_csub' ,   'ntu60_FDD_train_csub' ,  'ntu60_JDD_test_cview' , 'ucla_ADD_test'    ,      'ucla_JDD_train',
'ntu120_FDD_train_csetup' , 'ntu120_JDD_test_csub' ,   'ntu60_ADD_test_csub'   ,   'ntu60_FDD_train_cview' , 'ntu60_JDD_train_csub' , 'ucla_ADD_train'   ,      'ucla_LAD_test']

base_directory = './data/descriptors/'

for dir in list_folders:
    full_path = os.path.join(base_directory, dir)
    os.makedirs(full_path)

