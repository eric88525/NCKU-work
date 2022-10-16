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
#include <map>
#include <chrono>
#include <stack>

using namespace std;

// record the itemset ans it's support count
class assoInfo
{
public:
    // itemSet
    set<string> itemSet;
    // support count
    int appearCount;

    assoInfo() : itemSet({}), appearCount(0) {}
    assoInfo(set<string> itemSet, int appearCount) : itemSet(itemSet), appearCount(appearCount) {}
    assoInfo(const assoInfo &p1)
    {
        this->itemSet = p1.itemSet;
        this->appearCount = p1.appearCount;
    }
};

// generate all  possible {a}=>{b} patterns and print support & confidence
// given itemset { 1, 2, 3} , return all combination  of {} => {} ex. like {1}=>{2,3}
vector<pair<set<string>, set<string>>> allCombination(const set<string> &freqItemSet)
{
    vector<pair<set<string>, set<string>>> result;

    queue<pair<set<string>, set<string>>> que;

    vector<string> itemSet(freqItemSet.begin(), freqItemSet.end());

    que.push(make_pair(set<string>(), set<string>()));

    int index = 0;

    while (que.size() && index < itemSet.size())
    {

        int n = que.size();
        while (n--)
        {
            auto temp1 = que.front();
            auto temp2 = que.front();
            que.pop();

            temp1.first.insert(itemSet[index]);
            temp2.second.insert(itemSet[index]);

            que.push(temp1);
            que.push(temp2);
        }
        index++;
    }
    while (que.size())
    {
        if (que.front().first.size() && que.front().second.size())
            result.push_back(que.front());
        que.pop();
    }
    return result;
}
