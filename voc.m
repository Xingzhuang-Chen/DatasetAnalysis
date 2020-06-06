function dataset = voc(file_list_path)
[~, filename, ext] = fileparts(file_list_path);
filename = [filename, ext, '.mat'];
if exist(filename, 'file')
    disp('exist')
    load(filename)
    dataset = dataset;
    return
end
file_list = fileread(file_list_path);
file_list = strsplit(file_list);

class = cell(0);
data = cell(0);
for i = 1:length(file_list)
    file_name = file_list{i};
    if isempty(file_name)
        continue
    end
    dom = xmlread(fullfile('Annotations', sprintf('%s.xml', file_name)));
    sizeNode = dom.getElementsByTagName('size').item(0);
    width = str2double(sizeNode.getElementsByTagName('width').item(0).getTextContent);
    height = str2double(sizeNode.getElementsByTagName('height').item(0).getTextContent);
    area = width*height;
    data{i,1} = width;
    data{i,2} = height;
    data{i,3} = area;
    
    objectNodes = dom.getElementsByTagName('object');
    num_bbox = objectNodes.getLength;
    box = cell(num_bbox, 3);
    remove_idx = [];
    for j = 1:objectNodes.getLength
        boxNode = objectNodes.item(j-1);
        if boxNode.getElementsByTagName('name').getLength==0
            remove_idx = [remove_idx, j];
            continue
        end
        name = boxNode.getElementsByTagName('name').item(0).getTextContent.toCharArray';
        bboxNode = boxNode.getElementsByTagName('bndbox').item(0);
        xmin = str2double(bboxNode.getElementsByTagName('xmin').item(0).getTextContent);
        xmax = str2double(bboxNode.getElementsByTagName('xmax').item(0).getTextContent);
        ymin = str2double(bboxNode.getElementsByTagName('ymin').item(0).getTextContent);
        ymax = str2double(bboxNode.getElementsByTagName('ymax').item(0).getTextContent);
        area = (xmax - xmin) * (ymax - ymin);
        box{j, 1} = name;
        box{j, 2} = [xmin, ymin, xmax, ymax];
        box{j, 3} = area;
        if ~ismember(name, class)
            class = [class, {name}];
        end
    end
    box(remove_idx, :) = [];
    data{i, 4} = box;
end
dataset = {class, data};
save(filename, 'dataset')
end