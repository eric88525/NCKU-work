#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
using namespace std;

class treeNode
{

public:
    int item;
    treeNode *parent;
    unordered_map<int, treeNode *> childs;
    int count;

    treeNode()
    {
        this->parent = NULL;
        this->childs = unordered_map<int, treeNode *>();
        this->count = -1;
        this->item = -1;
    }
    treeNode(treeNode *parent, int item)
    {
        this->parent = parent;
        this->childs = unordered_map<int, treeNode *>();
        this->count = 1;
        this->item = item;
    }
};

class fpTree
{

private:
    treeNode *root;
    unordered_map<int, int> frequency;
    unordered_map<int, vector<treeNode *>> itemPointers;
    int minSup;

public:
    fpTree(int minSup) : minSup(minSup)
    {
        this->frequency = unordered_map<int, int>({});
        this->itemPointers = unordered_map<int, vector<treeNode *>>({});
        this->root = new treeNode();
    }
    void addNode(vector<int> transation)
    {
        treeNode *p = root;
        treeNode *parent = p;

        sort(transation.begin(), transation.end(), sortByFreq);

        for (auto item : transation)
        {
            parent = p;
            if (p->childs.count(item))
            {
                p->count++;
            }
            else
            {
                treeNode *newItem = new treeNode(parent, item);
                p->childs[item] = newItem;
                itemPointers[item].push_back(newItem);
            }
            p = p->childs[item];
        }
    }

    bool sortByFreq(int &a, int &b)
    {
        return this->frequency[a] > this->frequency[b];
    }
    void countItems(vector<int> transation)
    {
        for (auto &item : transation)
            this->frequency[item]++;
    }
    void clearNode()
    {
        return;
    }
};

vector<vector<int>> readData(string filePath)
{

    int id, tranID, item;

    ifstream infile(filePath);

    unordered_map<int, vector<int>> datas;

    while (infile >> id >> tranID >> item)
    {
        datas[tranID].push_back(item);
    }

    vector<vector<int>> result;

    for (auto &d : datas)
    {
        result.push_back(d.second);
    }
    return result;
}

int main()
{

    treeNode a(NULL, 1);

    // create tree
    fpTree tree(3);

    // read data
    vector<vector<int>> datas = readData("./datas.txt");

    // count items
    for (auto &d : datas)
        tree.countItems(d);

    // clear unfrequent item
    tree.clearNode();

    // create tree
    for (auto &d : datas)
        tree.addNode(d);
}