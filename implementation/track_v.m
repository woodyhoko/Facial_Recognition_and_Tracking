function [ppp] = track_v(etcsq,videoPlayer,im)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            im_to_show = double(im)/255;
            step(videoPlayer, im_to_show);
            %imagesc(im_to_show);
            hold on;
            for n=1:size(etcsq)
                im_to_show=insertObjectAnnotation(im_to_show, 'rectangle', etcsq{n}{1}, 'Face');
            end
            im_to_show=insertText(im_to_show,[10 10], num2str(1/toc));
            %rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
            
%             output_name = 'Video_name';
%             opengl software;
%             writer = VideoWriter(output_name, 'MPEG-4');
%             writer.FrameRate = 5;
%             open(writer);
    
        drawnow
%         if frame > 1
%             if frame < inf
%                 writeVideo(writer, getframe(gcf));
%             else
%                 close(writer);
%             end
%         end
%          pause
    end