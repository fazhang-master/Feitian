# -*- coding: utf-8 -*-

'''
更新：加入DLKcat
加入可视化界面
加入mask
'''


def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


import os
import sys
current = os.getcwd()
sys.path.append((os.path.join(current,'DPA')))
sys.path.append(os.path.join(current,'TurNuP\code'))
import time
import copy
import heapq
import random
import numpy as np
import pandas as pd

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np


import DPA.prediction_for_input as pre
from TurNuP.code.kcat_prediction import kcat_predicton

import esm

class MyWindow(QWidget):
    # 声明一个信号 只能放在函数外面
    my_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.msg_history = list() # 用来存放消息

    def init_ui(self):
        self.resize(1200, 1000)
        self.setWindowTitle('飞天')
        self.setWindowIcon(QIcon((os.path.join(current,'icons/graduation-cap.ico'))))

        # 创建一个整体布局器
        container = QVBoxLayout()

        # 用来显示检测到漏洞的信息
        self.msg =  QLabel("")
        self.msg.resize(1000, 15)
        # print(self.msg.frameSize())
        # self.msg.setWordWrap(True) # 自动换行
        self.msg.setAlignment(Qt.AlignTop) # 靠上
        # self.msg.setStyleSheet("background-color: yellow; color: black;")

        # 创建一个滚动对象
        scroll = QScrollArea()
        scroll.setWidget(self.msg)

        # 创建垂直布局器，用来添加自动滚动条
        v_layout = QVBoxLayout()
        v_layout.addWidget(scroll)

        # 创建水平布局器
        h_layout = QVBoxLayout()

        # Substrate Name
        self.label1 = QLabel('Substrate Name:', self)
        self.edit1 = QLineEdit(self)
        self.edit1.setPlaceholderText('Catechol')

        # Substrate SMILES
        self.label2 = QLabel('Substrate SMILES:', self)
        self.edit2 = QLineEdit(self)
        self.edit2.setPlaceholderText('C1=CC=C(C(=C1)O)O')

        # Product SMILES
        self.label3 = QLabel('Product SMILES:', self)
        self.edit3 = QLineEdit(self)
        self.edit3.setPlaceholderText('C1=CC(=CC=C1C(C(=O)O)N)O')

        # Protein Sequence
        self.label4 = QLabel('Protein Sequence:', self)
        self.edit4 = QTextEdit(self)
        self.edit4.setPlaceholderText('MVHVRKNHLTMTAEEKRR')

        # Mask
        self.label5 = QLabel('Mask:', self)
        self.edit5 = QLineEdit(self)
        self.edit5.setPlaceholderText('Mutations that can cause disruption, e.g:R3A T5D A44D...')

        # Output_name
        self.label6 = QLabel('Output Name', self)
        self.edit6 = QLineEdit(self)
        self.edit6.setPlaceholderText('e.g:demo')


        btn = QPushButton("submit", self)
        # 绑定按钮的点击，点击按钮则开始检测
        btn.clicked.connect(self.check)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label1)
        h_layout.addWidget(self.edit1)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label2)
        h_layout.addWidget(self.edit2)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label3)
        h_layout.addWidget(self.edit3)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label4)
        h_layout.addWidget(self.edit4)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label5)
        h_layout.addWidget(self.edit5)
        h_layout.addStretch(1)
        h_layout.addWidget(self.label6)
        h_layout.addWidget(self.edit6)
        h_layout.addStretch(1) # 伸缩器
        h_layout.addWidget(btn)
        h_layout.addStretch(1)

        # 操作将要显示的控件以及子布局器添加到container
        container.addLayout(h_layout)
        container.addLayout(v_layout)

        # 设置布局器
        self.setLayout(container)

        # 绑定信号和槽
        self.my_signal.connect(self.my_slot)

    def my_slot(self, msg):
        # 更新内容
        # print(msg)
        self.msg_history.append(msg)
        self.msg.setText('<br>'.join(self.msg_history))
        self.msg.resize(440,self.msg.frameSize().height() + 40)
        self.msg.repaint() # 更新内容，如果不更新可能没有显示新内容


    def check(self):
#
        start = time.time()
        AA = ['R', 'G', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'A', 'H']
        S_Name = self.edit1.text()
        S_SMILES = self.edit2.text()
        P_SMILES = self.edit3.text()
        space = "	"
        Sequence = self.edit4.toPlainText()

######################Tur##################################################
        substrates = list()
        products = list()
        enzymes = list()
        OriginAA = list()
        site = list()
        MutationAA = list()
######################Tur##################################################

        self.my_signal.emit('Begin construction of mutant libraries')
        title = 'Substrate Name	Substrate SMILES	Protein Sequence' + '\n'
        print(os.path.dirname(__file__))
        if os.path.exists(os.path.join(current,'DPA/input.tsv')):
            os.remove(os.path.join(current,'DPA/input.tsv'))
        with open(os.path.join(current,'DPA/input.tsv'), 'w', encoding='utf-8') as fileobj:
            fileobj.write(title)
        num_list = []   # [0,1,2]
        for i in range(len(Sequence)):
            for j in AA:
                print(i, j)

                ######################Tur##################################################
                OriginAA.append(Sequence[i])
                site.append(i + 1)
                MutationAA.append(j)
                ######################Tur##################################################

                num_list.append({i+1: (Sequence[i],j)})
                new_seq = self.replace_char(Sequence, j, i)

                ######################Tur##################################################
                substrates.append(S_SMILES)
                products.append(P_SMILES)
                enzymes.append(new_seq)
                ######################Tur##################################################

                with open(os.path.join(current,'DPA/input.tsv'), 'a+', encoding='utf-8') as fileobj:
                    str = S_Name + space + S_SMILES + space + new_seq + '\n'
                    print(str)
                    # self.my_signal.emit(str.replace('\n', ''))
                    fileobj.write(str)
                    # time.sleep(0.1)
        self.my_signal.emit('-----------------------------------')
        self.my_signal.emit('It\'s time to start the prediction!')
        self.my_signal.emit('Calculating numerical representations for all substrates and products.')
        self.my_signal.emit('Calculating numerical representations for all enzymes.')
        self.my_signal.emit('Loading ESM-1b model.')
        self.my_signal.emit('Loading model parameters for task-specific model.')
        self.my_signal.emit('Calculating enzyme representations.')
        # os.system('python prediction_for_input.py input.tsv')
        self.my_signal.emit('Please be patient, it is being calculated...')
        pre.main()
        time.sleep(2)

##################################################################################################################################
        info = {"OriginAA": OriginAA, "site": site, "MutationAA": MutationAA}
        Tur_df = kcat_predicton(substrates=substrates,
                            products=products,
                            enzymes=enzymes,
                            info=info)
        self.my_signal.emit('Making predictions for kcat.')
        if Tur_df is not None:
            Tur_df.to_csv(os.path.join(current,'TurNuP/code/TurNuP.tsv'), index=False)
        DPA_df_path = os.path.join(current, 'DPA/output.tsv')
        Tur_df_path = os.path.join(current, 'TurNuP/code/TurNuP.tsv')
        DPA_df = pd.read_csv(DPA_df_path, sep='\t')
        Tur_df = pd.read_csv(Tur_df_path, sep=',')
        self.out_file(DPA_df, Tur_df)
        self.my_signal.emit('-----------------------------------')
        self.my_signal.emit('Prediction Done!')
        self.my_signal.emit('-----------------------------------')
        self.my_signal.emit('output path:' + current)
###################################################################################################################################

        end = time.time()
        Elapsed_time = end - start
        print('消耗用时：%.2fs'%Elapsed_time)
        self.my_signal.emit('Elapsed_time:%.2fs'%Elapsed_time)

    def replace_char(self, old_string, char, index):
        '''
        字符串按索引位置替换字符
        '''
        return f'{old_string[:index]}{char}{old_string[index + 1:]}'

    def out_file(self, DPA_df, Tur_df):
        mask = self.edit5.text()
        out = self.edit6.text()
        out1_path = os.path.join(current, f'{out}1.csv')
        out2_path = os.path.join(current, f'{out}2.csv')
        DPA_df = DPA_df.rename(columns= {'Protein Sequence': 'enzyme','Kcat value (1/s)': 'K1'})
        Tur_df = Tur_df.rename(columns= {'kcat [s^(-1)]': 'K2'})
        DPA_df = DPA_df.round({'K1': 2})
        Tur_df = Tur_df.round({'K2': 2})
        DPA_Tur = pd.merge(DPA_df, Tur_df,  on=['enzyme'])
        DPA_Tur = DPA_Tur.drop(columns=['Substrate Name', 'Substrate SMILES', 'substrates', 'products', 'difference_fp', 'enzyme rep', ])
        order = ['K1', 'enzyme', 'OriginAA', 'site', 'MutationAA', 'complete', 'K2']
        DPA_Tur = DPA_Tur.reindex(columns= order)
        DPA_Tur.drop_duplicates(subset=['enzyme'], keep='first', inplace=True)
        mask = mask.split()
        for m in mask:
            # print(DPA_Tur[(DPA_Tur['OriginAA'] == m[0]) & (DPA_Tur['site'] == int(m[1])) & (DPA_Tur['MutationAA'] == m[2])].index)
            DPA_Tur.drop(DPA_Tur[(DPA_Tur['OriginAA'] == m[0]) & (DPA_Tur['site'] == int(m[1])) & (DPA_Tur['MutationAA'] == m[2])].index, inplace=True)
        DPA_Tur.to_csv(out1_path, index = False)
        indx = DPA_Tur.loc[DPA_Tur[(DPA_Tur['OriginAA'] == 'M') & (DPA_Tur['site'] == 1) & (DPA_Tur['MutationAA'] == 'M')].index]
        indx.drop(columns=['enzyme', 'OriginAA', 'site', 'MutationAA', 'complete'], inplace= True)
        id_a = list(indx.idxmax(axis=1))[0]
        max_a = float(list(indx.max(axis=1))[0])
        id_b = list(indx.idxmin(axis=1))[0]
        min_b = float(list(indx.min(axis=1))[0])
        DPA_Tur['New kcat'] = (max_a/min_b)*DPA_Tur[id_b]
        DPA_Tur['kcat [s^(-1)]'] = DPA_Tur[['K2', 'New kcat']].max(axis=1)
        DPA_Tur.drop(columns=['K1', 'K2', 'New kcat'], inplace= True)
        DPA_Tur = DPA_Tur.round({'kcat [s^(-1)]': 2})
        DPA_Tur.to_csv(out2_path, index = False)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    w.show()

    sys.exit(app.exec_())

