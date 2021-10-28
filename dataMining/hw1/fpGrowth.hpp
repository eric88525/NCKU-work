
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <set>
#include <queue>
#include <iomanip>
#include <unordered_map>

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

class assoInfo
{
public:
    //string item;
    set<string> itemSet;

    int support;

    assoInfo() : itemSet({}), support(0) {}
    assoInfo(const assoInfo &p1)
    {
        //  this->item = p1.item;
        this->itemSet = p1.itemSet;
        this->support = p1.support;
    }
};

class fpTree
{

public:
    static int transCount;
    treeNode *root;
    int minSup;

    map<string, int> frequency;

    // record last seen treeNode of item
    map<string, treeNode *> itemPointers;

    // list all item by freq low to hight
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
    void addNode(const vector<vector<string>> &transations)
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

        for (auto const &i : this->frequency)
        {
            string temp = i.first;
            allItems.push_back(temp);
        }

        sortByFreq(allItems, 0, allItems.size() - 1);
        // from low freq to high freq
        reverse(allItems.begin(), allItems.end());
        this->itemOrder = allItems;
    }

    // build map of < item : frequency> and create order
    void countItems(const vector<vector<string>> &datas)
    {
        map<string, int> temp;
        for (auto const &transation : datas)
        {
            for (auto const &item : transation)
                temp[item]++;
        }
        for (auto const &i : temp)
        {
            if (i.second >= this->minSup)
            {
                this->frequency[i.first] = i.second;
            }
        }
        createOrder();
    }
    // find freq itemset by limit
    vector<assoInfo> fpMining(assoInfo history = assoInfo())
    {

        vector<assoInfo> result;

        for (auto item : itemOrder)
        {
            // make a copy of history
            assoInfo newHistory = history;

            // add current item
            newHistory.itemSet.insert(item);
            newHistory.support = this->frequency[item];
            result.push_back(newHistory);

            // create cond tree by item
            treeNode *n = itemPointers[item];
            fpTree cTree = condTree(n);

            // recursive mining
            vector<assoInfo> temp = cTree.fpMining(newHistory);

            // add mingin result to global result
            result.insert(result.end(), temp.begin(), temp.end());
        }
        return result;
    }

    // create cond tree of item
    fpTree condTree(treeNode *n)
    {

        fpTree tree(minSup);

        vector<vector<string>> newTransations;

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