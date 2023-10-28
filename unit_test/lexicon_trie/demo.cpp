#include "lexicon_trie.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <memory>

using namespace std;
int main()
{  
    string unit_file = "/path/to/phoneme.txt";
    string lexicon_file = "/path/to/word_to_phoneme.txt";

    std::shared_ptr<asrdec::UnitTable> table = std::make_shared<asrdec::UnitTable>();
    std::shared_ptr<asrdec::lt::LexiconTrie> trie = std::make_shared<asrdec::lt::LexiconTrie>();
    
    table->build( unit_file );
    std::cout << "Build Table Done: " << table->size() << std::endl;

    trie->build( lexicon_file, table, "", false);

    std::cout << "Build Trie Done! Max deepth: " <<  trie->deepth() << std::endl;

    // for ( size_t i=0; i<table->size(); i++ )
    // {
    //     std::cout << i << " : " << table->sym(i) << std::endl;
    // }
    trie->save("/path/to/trie.txt");

    return 0;
}