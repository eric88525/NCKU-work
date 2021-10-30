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
#include <chrono>

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

// generate all  possible {a}=>{b} patterns and print support & confidence
// given itemset { 1, 2, 3} , return all combination  of {} => {} ex. like {1}=>{2,3}
vector<pair<string, string>> allCombination(const set<string> &freqItemSet)
{
    vector<pair<string, string>> result;

    queue<pair<string, string>> que;

    vector<string> itemSet(freqItemSet.begin(), freqItemSet.end());

    que.push(make_pair("", ""));

    int index = 0;

    while (que.size() && index < itemSet.size())
    {

        int n = que.size();
        while (n--)
        {
            auto temp1 = que.front();
            auto temp2 = que.front();
            que.pop();

            temp1.first += itemSet[index] + ", ";
            temp2.second += itemSet[index] + ", ";

            que.push(temp1);
            que.push(temp2);
        }
        index++;
    }
    while (que.size())
    {
        if (que.front().first != "" && que.front().second != "")
            result.push_back(que.front());
        que.pop();
    }
    return result;
}
