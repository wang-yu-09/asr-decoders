/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#pragma once
#include <string>
#include <vector>
#include <iostream>

namespace asrdec
{

/* 解码器参数
*  这个参数配置用于所有的解码器
*  有的解码器只使用相应的部分参数
*/
struct DecodeParams
{
    /* 每行的格式: [word] [probability] [unit1] [unit2] ... */
    std::string lexiconp_file;
    /* 每行的格式: [word] [units] */
    std::string lexicon_file;
    /* 每行的格式: [unit] */ 
    std::string unit_file;
    /* unk符号 */
    std::string unk_symbol = "<unk>";
    /* blank id */
    size_t blank_id = 0;
    /* 静音符号 */
    size_t silence_id = 0;
    /* 子词级语言模型及权重 */
    std::string unit_kenlm_file;
    float unit_lm_weight = 1.0;
    /* 声学模型剪枝beam size, 数量 */
    size_t unit_beam_size = 15;
    /* 路径剪枝beam size, 数量 */
    size_t beam_size = 15;
    /* 单词级语言模型及剪枝size */
    std::string word_kenlm_file;
    float word_lm_weight = 1.0;
    size_t word_beam_size = 15;
    /* 词间语言模型及权重 */
    std::string wwi_kenlm_file;
    float wwi_lm_weight = 1.0;
    /* 静音阈值,超过这个数量将会跳帧 */
    size_t max_silence_frames = 10;
    bool truncate_silence = false;
    /* 下面是一些实验性功能的参数 */
    bool callback_pruned_beam = false;
    

    std::string lm_prune_file;
    size_t context_prune_beam_size = 15;
    std::string hotword_file;
    std::string hotword_extra_lexicon_file;

};

/* 保存最终解码得到的结果
*/
struct DecodeResult
{
    /* CTC class id 序列 */
    std::vector<int> unit_ids;
    /* CTC class 单元序列 */
    std::vector<std::string> units;
    /* 单词序列(如果有) */
    std::vector<std::string> words;
    /* 声学模型得分(不含权重和正则化因子) */
    float am_score;
    /* 语言模型得分(不含权重和正则化因子) */
    float lm_score;

    public:
        DecodeResult():am_score(0.0),lm_score(0.0)
        {
            // unit_ids.clear();
            // units.clear();
            // words.clear();
        }

        void clear()
        {
            unit_ids.clear();
            units.clear();
            words.clear();
            am_score=0.0;
            lm_score=0.0;
        }

        void show()
        {
            std::cout << "==============Decode Result==============\n";
            std::cout << "ID: "; 
            for ( size_t i=0; i<unit_ids.size(); i++ )
            {
                std::cout << unit_ids[i] << " ";
            } 
            std::cout << "\n"; 
            std::cout << "UNITS: "; 
            for ( size_t i=0; i<units.size(); i++ )
            {
                std::cout << units[i] << " ";
            } 
            std::cout << "\n"; 
            std::cout << "WORDS: "; 
            for ( size_t i=0; i<words.size(); i++ )
            {
                std::cout << words[i] << " ";
            } 
            std::cout << "\n"; 
            std::cout << "=========================================\n";
        }
};

/* 保存最终解码得到的Nbest结果
*/
struct DecodeResultNbest
{
    /* CTC class id 序列 */
    std::vector<std::vector<int> > unit_ids;
    /* CTC class 单元序列 */
    std::vector<std::vector<std::string>> units;

    public:
        void show()
        {   
            for ( size_t t=0; t<unit_ids.size(); t++)
            {
                std::cout << "==============Decode Result==============\n";
                std::cout << "ID(" <<unit_ids[t].size() <<"): "; 
                for ( size_t i=0; i<unit_ids[t].size(); i++ )
                {
                    std::cout << unit_ids[t][i] << " ";
                } 
                std::cout << "\n"; 
                std::cout << "UNITS(" <<units[t].size() <<"): "; 
                for ( size_t i=0; i<units[t].size(); i++ )
                {
                    std::cout << units[t][i] << " ";
                } 
                std::cout << "\n"; 
                std::cout << "=========================================\n";
            }
        }    
};

} // namespace asrdec