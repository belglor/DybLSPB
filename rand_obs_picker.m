picked_idx = nan(1, 10); 
for cls = 0:9
    whereClsIsFound = find(lb == cls);
    theChosenOne = randi(length(whereClsIsFound));
    picked_idx(cls+1) = whereClsIsFound(theChosenOne);
end
picked_obs = ob_wav(picked_idx, :);
save('fold10_RANDOM_OBS', 'picked_obs')