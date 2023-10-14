function [adjGroupsij, adjGroupsInd] = fuserLassoAdjGroups(groups)
adjGroupsij = [];
adjGroupsInd = 1:size(groups, 1);
for iG = 1:size(groups, 1)
    for i = groups(iG, 1):groups(iG, 2)-1
        adjGroupsij(end+1, :) = [i, i+1];
    end
end