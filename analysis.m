clear all
close all

dataset = voc(fullfile('ImageSets', 'Main', 'all.txt'));
class = dataset{1};
data = dataset{2};

img_widths = [data{:, 1}];
img_heights = [data{:, 2}];
img_areas = [data{:, 3}];

% class_list = cell(1, length(class));
class_list = repmat({zeros(0,5)},1,5);
for i = 1:size(data, 1)
    boxs = data{i, 4};
    img_area = data{i, 3};
    for j = 1:size(boxs, 1)
        cls = boxs{j, 1};
        idx = find(ismember(class, cls ));
        bbox = boxs{j, 2};
        box_width = bbox(3) - bbox(1);
        box_height = bbox(4) - bbox(2);
        box_ratio = box_height/box_width;
        box_area = boxs{j, 3};
        box_scale = box_area/img_area;
        class_list{idx} = [class_list{idx}; [box_width, box_height, box_ratio, box_area, box_scale]];
    end
end

figure('Name', 'image')
subplot(2, 2, 1)
histogram(img_widths, 100)
title('image width')
subplot(2, 2, 2)
histogram(img_heights, 100)
title('image height')
subplot(2, 1, 2)
histogram(img_areas, 100)
title('image area')

ratio = [];
scale = [];
for i = 1:length(class)
    figure('Name', class{i})
    subplot(1, 2, 1)
    ratio = [ratio; class_list{i}(:, 3)];
    histogram(class_list{i}(:, 3), 100);
    title('box ratio');
    subplot(1, 2, 2)
    scale = [scale; class_list{i}(:, 5)];
    histogram(class_list{i}(:, 5), 100);
    title('box scale(box area/img area)');
end
figure('Name', 'All bbox')
subplot(1, 2, 1)
histogram(ratio, 0:0.05:5);
title('box ratio');
subplot(1, 2, 2)
histogram(scale, 0:0.01:0.5);
title('box scale(box area/img area)');

original_wh = 1024;
scale = scale*original_wh^2;

anchor_scale = sqrt(scale)*(1./[4, 8, 16, 32, 64]);
figure('Name', 'All stage anchor scale');
hold on
histogram(anchor_scale(:,1), (0:0.001:0.15)*original_wh)
histogram(anchor_scale(:,2), (0:0.001:0.15)*original_wh)
histogram(anchor_scale(:,3), (0:0.001:0.15)*original_wh)
histogram(anchor_scale(:,4), (0:0.001:0.15)*original_wh)
histogram(anchor_scale(:,5), (0:0.001:0.15)*original_wh)
title('All stage anchor scale');


fprintf('original box scale(h,w) mean= %f\n', sqrt(mean(scale)));
fprintf('stage1 box scale(h,w) mean= %f\n', mean(anchor_scale(:,1)));
fprintf('stage2 box scale(h,w) mean= %f\n', mean(anchor_scale(:,2)));
fprintf('stage3 box scale(h,w) mean= %f\n', mean(anchor_scale(:,3)));
fprintf('stage4 box scale(h,w) mean= %f\n', mean(anchor_scale(:,4)));
fprintf('stage5 box scale(h,w) mean= %f\n', mean(anchor_scale(:,5)));
