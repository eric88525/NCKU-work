#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>

#include <queue>

using namespace std;

class treeNode
{

public:
    string item;
    treeNode *parent;
    map<string, treeNode *> childs;
    int count;
    treeNode *nextHomonym;
    treeNode()
    {
        this->parent = NULL;
        this->count = -1;
        this->item = "root";
        this->nextHomonym = NULL;
    }
    treeNode(treeNode *parent, string item)
    {
        this->parent = parent;
        this->count = 1;
        this->item = item;
        this->nextHomonym = NULL;
    }
};

class fpTree
{

public:
    treeNode *root;
    int minSup;

    map<string, int> frequency;
    // record previous node of item
    map<string, treeNode *> itemPointers;

    vector<string> itemOrder;

    fpTree(int minSup) : minSup(minSup)
    {
        this->frequency = {};
        this->itemPointers = {};
        this->itemOrder = {};
        this->root = new treeNode();
    }
    // quick sort
    void sortByFreq(vector<string> &transation, int low, int high)
    {
        if (low < high)
        {
            int pi = partition(transation, low, high);
            sortByFreq(transation, low, pi - 1);
            sortByFreq(transation, pi + 1, high);
        }
    }

    int partition(vector<string> &arr, int low, int high)
    {
        int pivot = frequency[arr[high]];
        int i = low - 1;

        for (int j = low; j < high; j++)
        {
            if (frequency[arr[j]] > pivot)
            {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }
    vector<string> removeUnFreq(const vector<string> &transation)
    {
        vector<string> result;
        for (auto t : transation)
        {
            if (frequency.count(t) && frequency[t] >= minSup)
                result.push_back(t);
        }
        return result;
    }
    void addNode(vector<vector<string>> transations)
    {
        for (auto transation : transations)
        {
            treeNode *p = root;
            treeNode *parent = p;

            transation = removeUnFreq(transation);

            sortByFreq(transation, 0, transation.size() - 1);

            for (auto item : transation)
            {

                if (p->childs.count(item))
                {
                    p = p->childs[item];
                    p->count++;
                }
                else
                {
                    // create new node
                    parent = p;
                    treeNode *newItem = new treeNode(parent, item);

                    // let nextHomonym point to previous item
                    if (itemPointers.count(item))
                    {
                        newItem->nextHomonym = itemPointers[item];
                    }
                    itemPointers[item] = newItem;

                    // go to child
                    p->childs[item] = newItem;
                    p = p->childs[item];
                }
            }
        }
    }
    // sort all item from high freq to low freq & remove unfreq item
    void createOrder()
    {
        vector<string> allItems;

        for (auto i : frequency)
        {
            string temp = i.first;
            allItems.push_back(temp);
        }

        sortByFreq(allItems, 0, allItems.size() - 1);
        itemOrder = allItems;
    }

    // build map of < item : frequency> and create order
    void countItems(vector<vector<string>> datas)
    {
        map<string, int> temp;
        for (auto const &transation : datas)
        {
            for (auto const &item : transation)
                temp[item]++;
        }
        for (auto i : temp)
        {
            if (i.second >= minSup)
            {
                frequency[i.first] = i.second;
            }
        }
        createOrder();
    }
    // find freq itemset by limit
    void fpMining(float confident, float support)
    {
        for (auto item : itemOrder)
        {
            fpTree tree = condTree(item);
        }
    }
    // create cond tree of item
    fpTree condTree(string item)
    {
        fpTree tree(minSup);

        vector<vector<string>> newTransations;

        treeNode *n = itemPointers[item];

        while (n)
        {
            int pCount = n->count;

            vector<string> preNodes;

            treeNode *p = n->parent;

            while (p->parent)
            {
                preNodes.push_back(p->item);
                p = p->parent;
            }
            while (pCount--)
            {
                newTransations.push_back(preNodes);
            }
            n = n->nextHomonym;
        }
        tree.countItems(newTransations);
        tree.addNode(newTransations);
        return tree;
    }
};

// get all transations
vector<vector<string>> readData(string filePath)
{

    int id, tranID;
    string item;

    ifstream infile(filePath);

    map<int, vector<string>> datas;

    while (infile >> id >> tranID >> item)
    {
        datas[tranID].push_back(item);
    }

    vector<vector<string>> result;

    for (auto &d : datas)
    {
        result.push_back(d.second);
    }
    return result;
}

void showTree(fpTree &tree)
{
    queue<treeNode *> list;

    treeNode *root = tree.root;
    list.push(root);
    cout << "[root]";
    while (list.size())
    {

        int n = list.size();

        cout << "\n=====================\n";
        while (n--)
        {
            treeNode *curr = list.front();
            list.pop();
            for (auto item : curr->childs)
            {
                cout << "[" << item.first << "," << item.second->count << "] ";
                list.push(item.second);
            }
        }
    }

    for (auto i : tree.itemOrder)
    {
        cout << i << " freq= " << tree.frequency[i] << "\n";
    }
}

int main()
{

    // create tree
    fpTree tree(2);

    // read data
    //vector<vector<string>> datas = readData("./data.txt");

    vector<vector<string>> datas = {
        {"milk", "bread", "beer"},
        {"bread", "coffee"},
        {"bread", "egg"},
        {"milk", "bread", "coffee"},
        {"milk", "egg"},
        {"bread", "egg"},
        {"milk", "egg"},
        {"milk", "bread", "egg", "beer"},
        {"milk", "bread", "egg"},
    };

    // count items
    tree.countItems(datas);

    // create tree
    tree.addNode(datas);

    float confident = 0.5, support = 1;

    showTree(tree);

    fpTree t2 = tree.condTree("milk");
    showTree(t2);
}