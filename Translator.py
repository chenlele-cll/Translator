# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Translator.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from nmt.nmt import create_or_load_hparams
from nmt.inference import get_model_creator
from nmt.model_helper import create_infer_model
from nmt.inference import start_sess_and_load_model
from nmt.inference import single_worker_inference
import tensorflow as tf
import jieba
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def creat_hparams(hmap):
    return tf.contrib.training.HParams(
        # Data
        src=hmap['src'],
        tgt=hmap['tgt'],
        train_prefix=hmap['train_prefix'],
        dev_prefix=hmap['dev_prefix'],
        test_prefix=hmap['test_prefix'],
        vocab_prefix=hmap['vocab_prefix'],
        embed_prefix=hmap['embed_prefix'],
        out_dir=hmap['out_dir'],

        # Networks
        num_units=hmap['out_dir'],
        num_encoder_layers=hmap['num_encoder_layers'],
        num_decoder_layers=hmap['num_decoder_layers'],
        dropout=hmap['dropout'],
        unit_type=hmap['unit_type'],
        encoder_type=hmap['encoder_type'],
        residual=hmap['residual'],
        time_major=hmap['time_major'],
        num_embeddings_partitions=hmap['num_embeddings_partitions'],

        # Attention mechanisms
        attention=hmap['attention'],
        attention_architecture=hmap['attention_architecture'],
        output_attention=hmap['output_attention'],
        pass_hidden_state=hmap['pass_hidden_state'],

        # Train
        optimizer=hmap['optimizer'],
        num_train_steps=hmap['num_train_steps'],
        batch_size=hmap['batch_size'],
        init_op=hmap['init_op'],
        init_weight=hmap['init_weight'],
        max_gradient_norm=hmap['max_gradient_norm'],
        learning_rate=hmap['learning_rate'],
        warmup_steps=hmap['warmup_steps'],
        warmup_scheme=hmap['warmup_scheme'],
        decay_scheme=hmap['decay_scheme'],
        colocate_gradients_with_ops=hmap['colocate_gradients_with_ops'],
        num_sampled_softmax=hmap['num_sampled_softmax'],

        # Data constraints
        num_buckets=hmap['num_buckets'],
        max_train=hmap['max_train'],
        src_max_len=hmap['src_max_len'],
        tgt_max_len=hmap['tgt_max_len'],

        # Inference
        src_max_len_infer=hmap['src_max_len_infer'],
        tgt_max_len_infer=hmap['tgt_max_len_infer'],
        infer_batch_size=hmap['infer_batch_size'],

        # Advanced inference arguments
        infer_mode=hmap['infer_mode'],
        beam_width=hmap['beam_width'],
        length_penalty_weight=hmap['length_penalty_weight'],
        coverage_penalty_weight=hmap['coverage_penalty_weight'],
        sampling_temperature=hmap['sampling_temperature'],
        num_translations_per_input=hmap['num_translations_per_input'],

        # Vocab
        sos=hmap['sos'],
        eos=hmap['eos'],
        subword_option=hmap['subword_option'],
        check_special_token=hmap['check_special_token'],
        use_char_encode=hmap['use_char_encode'],

        # Misc
        forget_bias=hmap['forget_bias'],
        num_gpus=hmap['num_gpus'],
        epoch_step=hmap['epoch_step'],  # record where we were within an epoch.
        steps_per_stats=hmap['steps_per_stats'],
        steps_per_external_eval=hmap['steps_per_external_eval'],
        share_vocab=hmap['share_vocab'],
        metrics=hmap['metrics'],
        log_device_placement=hmap['log_device_placement'],
        random_seed=hmap['random_seed'],
        override_loaded_hparams=hmap['override_loaded_hparams'],
        num_keep_ckpts=hmap['num_keep_ckpts'],
        avg_ckpts=hmap['avg_ckpts'],
        language_model=hmap['language_model'],
        num_intra_threads=hmap['num_intra_threads'],
        num_inter_threads=hmap['num_inter_threads'],
    )

map = {'attention':'scaled_luong',
       'attention_architecture':'standard',
       'avg_ckpts':False,
       'batch_size':300,
       'beam_width':0,
       'check_special_token':True,
       'colocate_gradients_with_ops':True,
       'coverage_penalty_weight':0.0,
       'decay_scheme':'',
       'dev_prefix':'nmt/tmp/nmt_data/dev',
       'dropout':0.2,
       'embed_prefix':'nmt/tmp/nmt_data/word2vec_512',
       'encoder_type':'uni',
       'eos':'</s>',
       'epoch_step': 0,
       'forget_bias':1.0,
       'infer_batch_size':32,
       'infer_mode':'greedy',
       'init_op':'uniform',
       'init_weight':0.1,
       'language_model':False,
       'learning_rate':1.0,
       'length_penalty_weight':0.0,
       'log_device_placement':False,
       'max_gradient_norm':5.0,
       'max_train':0,
       'metrics':['bleu'],
       'num_buckets':5,
       'num_decoder_layers':2,
       'num_embeddings_partitions':0,
       'num_encoder_layers':2,
       'num_gpus':1,
       'num_inter_threads':0,
       'num_intra_threads':0,
       'num_keep_ckpts':5,
       'num_sampled_softmax':0,
       'num_train_steps':300000,
       'num_translations_per_input':1,
       'num_units':512,
       'optimizer':'sgd',
       # 'out_dir':'nmt/tmp/nmt_attention_model/model_512_512_60000_300_r',
       'out_dir': 'nmt/tmp/nmt_attention_model/model_512_512_60000_300_ch_to_en',
       'output_attention':True,
       'override_loaded_hparams':False,
       'pass_hidden_state':True,
       'random_seed':None,
       'residual':False,
       'sampling_temperature':0.0,
       'share_vocab':False,
       'sos':'<s>',
       'src':'ch',
       'src_max_len':50,
       'src_max_len_infer': None,
       'steps_per_external_eval':None,
       'steps_per_stats':100,
       'subword_option':'',
       'test_prefix':'nmt/tmp/nmt_data/test',
       'tgt':'en',
       'tgt_max_len':50,
       'tgt_max_len_infer':None,
       'time_major':True,
       'train_prefix':'nmt/tmp/nmt_data/train',
       'unit_type':'lstm',
       'use_char_encode':False,
       'vocab_prefix':'nmt/tmp/nmt_data/vocab_noVec',
       'warmup_scheme':'t2t',
       'warmup_steps':0
       }
default_hparams1 = creat_hparams(map)
map['src'] = 'en'
map['tgt'] = 'ch'
map['batch_size'] = 256
map['out_dir'] = 'nmt/tmp/nmt_attention_model/model_512_512_60000_256_en_to_ch'
default_hparams2 = creat_hparams(map)

# out_dir1 = 'nmt/tmp/nmt_attention_model/model_512_512_60000_300_r'
# out_dir2 = 'nmt/tmp/nmt_attention_model/model_512_512_60000_300'
out_dir1 = 'nmt/tmp/nmt_attention_model/model_512_512_60000_300_ch_to_en'
out_dir2 = 'nmt/tmp/nmt_attention_model/model_512_512_60000_256_en_to_ch'

ckpt_path1 = tf.train.latest_checkpoint(out_dir1)
ckpt_path2 = tf.train.latest_checkpoint(out_dir2)
print(ckpt_path1)
print(ckpt_path2)

hparams1 = create_or_load_hparams(
    out_dir1, default_hparams1, None,
    save_hparams=0)

hparams2 = create_or_load_hparams(
    out_dir2, default_hparams2, None,
    save_hparams=0)

hparams1.inference_indices = None
hparams2.inference_indices = None

model_creator1 = get_model_creator(hparams1)
model_creator2 = get_model_creator(hparams2)

infer_model1 = create_infer_model(model_creator1, hparams1, None)
infer_model2 = create_infer_model(model_creator2, hparams2, None)

sess1, loaded_infer_model1 = start_sess_and_load_model(infer_model1, ckpt_path1)
sess2, loaded_infer_model2 = start_sess_and_load_model(infer_model2, ckpt_path2)

jieba.load_userdict("字典.txt")

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def split_word(data):
    sentence_list = sent_tokenize(data)
    res = ""

    for sentence in sentence_list:
        word_list = word_tokenize(sentence)
        res = res + " ".join(word_list)
        res = res + "\n"

    return res

class Ui_MainWindow(object):
    def __init__(self,MainWindow):
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)

        self.input_txt.textChanged.connect(self.textChanging_input_txt)
        self.execute_button.clicked.connect(self.clickButton_translate)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 618)
        MainWindow.setMinimumSize(QtCore.QSize(800, 618))
        MainWindow.setMaximumSize(QtCore.QSize(800, 618))

        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(15)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.input_txt = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.input_txt.setGeometry(QtCore.QRect(20, 119, 361, 411))
        self.input_txt.setObjectName("input_txt")
        self.input_txt.setFont(font)

        self.translate_res = QtWidgets.QTextBrowser(self.centralwidget)
        self.translate_res.setGeometry(QtCore.QRect(419, 119, 361, 411))
        self.translate_res.setObjectName("translate_res")
        self.translate_res.setFont(font)

        widgets_font = QtGui.QFont()
        widgets_font.setFamily("微软雅黑")
        widgets_font.setPointSize(12)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 10)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.src = QtWidgets.QLabel(self.centralwidget)
        self.src.setGeometry(QtCore.QRect(320, 52, 71, 31))
        self.src.setObjectName("src")
        self.src.setFont(widgets_font)
        self.horizontalLayout.addWidget(self.src)

        self.tgt = QtWidgets.QLabel(self.centralwidget)
        self.tgt.setGeometry(QtCore.QRect(430, 52, 71, 31))
        self.tgt.setObjectName("tgt")
        self.tgt.setFont(widgets_font)
        self.horizontalLayout.addWidget(self.tgt)

        self.detect_lang = QtWidgets.QLabel(self.centralwidget)
        self.detect_lang.setGeometry(QtCore.QRect(30, 70, 81, 31))
        self.detect_lang.setObjectName("detect_lang")
        self.detect_lang.setFont(widgets_font)

        self.tran_res = QtWidgets.QLabel(self.centralwidget)
        self.tran_res.setGeometry(QtCore.QRect(681, 80, 81, 20))
        self.tran_res.setObjectName("tran_res")
        self.tran_res.setFont(widgets_font)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("change.png"),QtGui.QIcon.Normal,QtGui.QIcon.Off)

        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(360, 45, 71, 51))
        self.toolButton.setObjectName("toolButton")
        self.toolButton.setIcon(icon)
        self.toolButton.setIconSize(QtCore.QSize(70, 50))
        self.toolButton.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton.setAutoRaise(False)
        self.toolButton.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton.setAutoFillBackground(True)
        self.toolButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton.setStyleSheet("border:none;")
        self.horizontalLayout.addWidget(self.toolButton)

        self.execute_button = QtWidgets.QPushButton(self.centralwidget)
        self.execute_button.setGeometry(QtCore.QRect(670, 540, 93, 28))
        self.execute_button.setObjectName("execute_button")
        self.execute_button.setFont(widgets_font)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "机器翻译-陈乐乐"))
        self.src.setText(_translate("MainWindow", "英文"))
        self.tgt.setText(_translate("MainWindow", "中文"))
        self.detect_lang.setText(_translate("MainWindow", "检测语言"))
        self.tran_res.setText(_translate("MainWindow", "翻译结果"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.execute_button.setText(_translate("MainWindow", "翻译"))

    def textChanging_input_txt(self):
        input_data = self.input_txt.toPlainText()

    def clickButton_translate(self):
        # 获取输入数据
        input_data = self.input_txt.toPlainText().strip()

        # 检测输入数据语言类型
        lang = is_contains_chinese(input_data)
        if lang:
            self.src.setText("中文")
            self.tgt.setText("英文")
            input_list = jieba.cut(input_data)
            input_data = " ".join(input_list)
        else:
            sentence_list = sent_tokenize(input_data)
            input_data = ""
            for sentence in sentence_list:
                word_list = word_tokenize(sentence)
                input_data += " ".join(word_list)
                input_data += "\n"
            # input_list = word_tokenize(input_data)
            # input_data = " ".join(input_list)
            self.tgt.setText("中文")
            self.src.setText("英文")

        # 写入输入数据
        with open("input_file.txt",'w',encoding='utf8') as fw:
            fw.seek(0)
            fw.truncate()
            fw.write(input_data)
            # input_list = input_data.split('.')
            # input_data = ""
            # for i in input_list:
            #     input_data = input_data + i + '.' + '\n'
            # fw.write(input_data)

        # 进行推导
        if lang:
            single_worker_inference(sess1,
                                    infer_model1,
                                    loaded_infer_model1,
                                    "input_file.txt",
                                    "output_file.txt",
                                    hparams1)
        else :
            single_worker_inference(sess2,
                                    infer_model2,
                                    loaded_infer_model2,
                                    "input_file.txt",
                                    "output_file.txt",
                                    hparams2)

        with open("output_file.txt","r",encoding='utf8') as fr:
            output_data = fr.read()
            if not lang:
                tmp = output_data.split(" ")
                output_data = "".join(tmp)
            self.translate_res.setText(output_data)
