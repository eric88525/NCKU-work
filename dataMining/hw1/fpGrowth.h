#pragma once
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

// record the itemset ans it's support count
class assoInfo
{
public:
    // itemSet
    set<string> itemSet;
    // support count
    int support;

    assoInfo() : itemSet({}), support(0) {}
    assoInfo(set<string> itemSet, int support) : itemSet(itemSet), support(support) {}
    assoInfo(const assoInfo &p1)
    {
        this->itemSet = p1.itemSet;
        this->support = p1.support;
    }
};

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

    // the item frequency
    map<string, int> frequency;

    // item node pointer
    map<string, treeNode *> itemPointers;

    // all item (freq high to low)
    vector<string> itemOrder;

    // init
    fpTree(int minSup)
    {
        this->minSup = minSup;
        this->frequency = {};
        this->itemPointers = {};
        this->itemOrder = {};
        this->root = new treeNode();
    }

    // quick sort by freq (high to low , string cmp high to low if freq[a]==freq[b])
    void sortByFreq(vector<string> &transation, int low, int high);
    int partition(vector<string> &arr, int low, int high);

    // remove item if it's support < minSupport
    void removeUnFreq(vector<string> &transation);

    // add tree node
    void addNode(const vector<vector<string>> &transations);

    // list all item from high freq to low freq & remove unfreq item
    void createOrder();

    // build map of  {item : frequency} and create order
    void buildTree(const vector<vector<string>> &datas);

    // return all freq itemset
    vector<assoInfo> fpMining(assoInfo history);

    // create cond tree of item
    fpTree condTree(string item);
};

// show tree structure
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

// quick sort by freq (high to low , string cmp high to low if freq[a]==freq[b])
void fpTree::sortByFreq(vector<string> &transation, int low, int high)
{
    if (low < high)
    {
        int pi = partition(transation, low, high);
        sortByFreq(transation, low, pi - 1);
        sortByFreq(transation, pi + 1, high);
    }
}

// quick sort by freq (high to low , string cmp high to low if freq[a]==freq[b])
int fpTree::partition(vector<string> &arr, int low, int high)
{
    int pivot = frequency[arr[high]];
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        if (this->frequency[arr[j]] > pivot || (this->frequency[arr[j]] == pivot && arr[j].compare(arr[high]) == 1))
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// remove item if it's support < minSupport
void fpTree::removeUnFreq(vector<string> &transation)
{
    vector<string> result;
    for (auto t : transation)
    {
        if (this->frequency.count(t) && this->frequency[t] >= minSup)
            result.push_back(t);
    }
    transation = result;
}

// add tree node
void fpTree::addNode(const vector<vector<string>> &transations)
{
    for (auto transation : transations)
    {
        treeNode *p = root;
        treeNode *parent = p;

        removeUnFreq(transation);

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

// list all item from high freq to low freq & remove unfreq item
void fpTree::createOrder()
{
    vector<string> allItems;

    for (const auto &i : this->frequency)
    {
        string temp = i.first;
        allItems.push_back(temp);
    }

    sortByFreq(allItems, 0, allItems.size() - 1);

    this->itemOrder = allItems;
}

// build map of  {item : frequency} and create order
void fpTree::buildTree(const vector<vector<string>> &datas)
{

    map<string, int> temp;

    //count item support count
    for (const auto &transation : datas)
    {
        for (const auto &item : transation)
        {
            temp[item]++;
        }
    }

    this->frequency.clear();

    // add freq one item to frequency
    for (auto i : temp)
    {
        if (i.second >= this->minSup)
        {
            this->frequency[i.first] = i.second;
        }
    }
    createOrder();
    addNode(datas);
}

// find freq itemset
vector<assoInfo> fpTree::fpMining(assoInfo history)
{

    vector<assoInfo> result;
    // item order is from high to low , so reverse it
    reverse(this->itemOrder.begin(), this->itemOrder.end());

    for (auto const item : this->itemOrder)
    {
        // make a copy of history
        assoInfo newHistory = history;

        // add current item
        newHistory.itemSet.insert(item);

        newHistory.support = this->frequency[item];

        result.push_back(newHistory);

        // create cond tree by item
        fpTree cTree = condTree(item);

        // recursive mining
        vector<assoInfo> temp = cTree.fpMining(newHistory);

        // add mining result to global result
        result.insert(result.end(), temp.begin(), temp.end());
    }
    return result;
}

// create cond tree of item
fpTree fpTree::condTree(string item)
{

    treeNode *n = itemPointers[item];
    fpTree tree(this->minSup);

    vector<vector<string>> newTransations;

    while (n)
    {
        int pCount = n->count;

        // parent nodes
        vector<string> preNodes;

        treeNode *p = n->parent;

        while (p->item != "root")
        {
            preNodes.push_back(p->item);
            p = p->parent;
        }

        if (preNodes.size())
        {
            while (pCount--)
            {
                newTransations.push_back(preNodes);
            }
        }

        n = n->nextHomonym;
    }

    tree.buildTree(newTransations);

    return tree;
}