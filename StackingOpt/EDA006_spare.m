
for mm=6:12
    mesh_range=0.02;
    if(mm == 2)
        ddmax=28;
    elseif(mm == 4 || mm == 6 || mm == 9 || mm == 11)
        ddmax=30;
    else
        ddmax=31;
    end
    if(mm < 8)
        yy = 14;
    else
        yy = 13;
    end
    if(mm == 8)
        ddstart = 15
    else
        ddstart = 1
   end
    for dd=ddstart:ddmax
        for tt = 0:27
            tt2 = tt * 0.5 +5;
            %メッシュ化したデータ読み込み
            %filename1="F:/study/preprocessing_data/3_mesh_place/%d月/%d月%d日/NV%d%d%d%.1f_time_sort.csv";
            filename1="F:/study/output/StackingOpt/EDA006/%d月/%d月%d日/Pred%d%d%d%.1f_time_sort2.csv";
            filename1=sprintf(filename1,mm,mm,dd,yy,mm,dd,tt2);
            disp(filename1);
            data_read=load(filename1);
            %30分分のデータをCSVファイルに作成するプログラム
            x=rmmissing(data_read);
            [xi,yi]=meshgrid(31.20:0.02:39.8,129.6:0.02:141.6);
            zi=griddata(x(:,1),x(:,2),x(:,3),xi,yi);
            %filename='F:/study/preprocessing_data/3_mesh_place/NV%d%.2f%.1f_zenkoku.csv';
            filename='H:/study/output/StackingOpt/EDA006/Pred%d%.2f%.1f_zenkoku2.csv';
            %filename1='F:/study/preprocessing_data/3_mesh_place/NV%d%.2f_int.csv';
            filename1='H:/study/output/StackingOpt/EDA006/Pred%d%.2f_int2.csv';
            %fp1='F:/study/preprocessing_data/3_mesh_place/long_zenkoku.csv';
            fp1='H:/study/output/StackingOpt/EDA006/long_zenkoku2.csv';
            %fp2='F:/study/preprocessing_data/3_mesh_place/lati_zenkoku.csv';
            fp2='H:/study/output/StackingOpt/EDA006/lati_zenkoku2.csv';
            filename=sprintf(filename,dd,mesh_range);
            filename1=sprintf(filename1,dd,mesh_range);
            fp1=sprintf(fp1);
            fp2=sprintf(fp2);
            fp=fopen(filename,'w');
            fplati=fopen(fp1,'w');
            fplongi=fopen(fp2,'w');
            fp3=fopen(filename1,'w');
            csvwrite(filename,zi);
            csvwrite(filename1,zi);
            csvwrite(fp1,yi);
            csvwrite(fp2,xi);
            fclose(fplati);
            fclose(fplongi);
            [xi,yi]=meshgrid(31.20:0.02:39.8, 129.6:0.02:141.6);
            zi=griddata(x(:,1),x(:,2),x(:,3),xi,yi,'natural');
            filename='H:/study/output/StackingOpt/EDA006/%d月/%d月%d日/Pred_interpolated_%d%d%d%.1f%.2f_int2.csv';
            filename=sprintf(filename,mm,mm,dd,yy,mm,dd,tt2,mesh_range);
            fp=fopen(filename,'w');
            csvwrite(filename,zi);
            fclose(fp);
            fclose('all');

        end
    fclose('all')
    end
clear all
end